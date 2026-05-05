import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
import tempfile
import os
import traceback
import noisereduce as nr

logger = logging.getLogger(__name__)

# Audio constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_DURATION = 3   # seconds
MAX_DURATION = 120  # seconds
SAMPLE_RATE = 22050  # Standard for speech analysis
N_MFCC = 40          # 40 coefficients for depression/stress models
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")

# VAD thresholds
VAD_THRESHOLD_DB    = -35.0   # frames below this dB are considered silent
VAD_MIN_SPEECH_RATIO = 0.20   # at least 20% of frames must be active speech
VAD_MIN_ACTIVE_DB   = -28.0   # mean dB of active frames must exceed this

# Spectral flatness thresholds
# Codec noise / silence: flatness close to 1.0 (energy spread flat across all bins)
# Real speech:           flatness well below 0.3 (energy concentrated in formants)
FLATNESS_MAX_MEAN   = 0.60   # reject if mean flatness exceeds this
FLATNESS_MIN_VOICED = 0.10   # min fraction of frames that look "voiced" (flatness < 0.3)


def validate_audio_file(audio_bytes: bytes, filename: str) -> Optional[str]:
    if len(audio_bytes) > MAX_FILE_SIZE:
        return f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"
    if len(audio_bytes) < 100:
        return "Audio file appears to be empty"
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
    return None


def _detect_voice_activity(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Frame-level energy VAD.

    Returns
    -------
    speech_ratio   : fraction of frames with active speech (0.0 – 1.0)
    mean_active_db : mean dB level of active frames (-80.0 if none found)
    """
    frame_length = int(sr * 0.025)  # 25 ms
    hop_length   = int(sr * 0.010)  # 10 ms

    rms    = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-4))  # floor at -80 dB

    active_frames  = rms_db > VAD_THRESHOLD_DB
    speech_ratio   = float(np.mean(active_frames))
    mean_active_db = float(np.mean(rms_db[active_frames])) if np.any(active_frames) else -80.0

    logger.info(
        "VAD: speech_ratio=%.3f, mean_active_db=%.1fdB, threshold=%.1fdB, "
        "total_frames=%d, active_frames=%d",
        speech_ratio, mean_active_db, VAD_THRESHOLD_DB,
        len(rms_db), int(np.sum(active_frames)),
    )
    return speech_ratio, mean_active_db


def _check_spectral_flatness(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Spectral flatness distinguishes real speech from codec noise / silence.

    Codec noise / silence → flatness ≈ 0.8–1.0  (energy uniformly spread)
    Real speech           → flatness ≈ 0.05–0.25 (energy in narrow formant bands)

    Returns
    -------
    mean_flatness   : mean spectral flatness across all frames (0.0 – 1.0)
    voiced_ratio    : fraction of frames with flatness < 0.3 (speech-like)
    """
    hop_length = int(sr * 0.010)  # 10 ms hop matches VAD

    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

    mean_flatness = float(np.mean(flatness))
    voiced_ratio  = float(np.mean(flatness < 0.3))

    logger.info(
        "Spectral flatness: mean=%.3f, voiced_ratio=%.3f "
        "(speech typically mean<0.3, voiced_ratio>0.10)",
        mean_flatness, voiced_ratio,
    )
    return mean_flatness, voiced_ratio


def process_audio_file(
    audio_bytes: bytes, filename: Optional[str] = None
) -> Tuple[Dict[str, np.ndarray], float]:

    temp_file = None
    try:
        logger.info(
            "Starting audio processing, file size: %s bytes, filename: %s",
            len(audio_bytes), filename,
        )

        # Preserve the original extension so audioread/ffmpeg can decode
        # browser-recorded uploads like .webm correctly.
        suffix = ".ogg"
        if filename:
            candidate_suffix = Path(filename).suffix.lower()
            if candidate_suffix in ALLOWED_EXTENSIONS:
                suffix = candidate_suffix

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(audio_bytes)
        temp_file.close()
        logger.info("Temporary file created: %s", temp_file.name)

        # ── Load ──────────────────────────────────────────────────────────────
        logger.info("Loading audio with librosa...")
        y, sr = librosa.load(temp_file.name, sr=SAMPLE_RATE, mono=True)
        logger.info("Audio loaded: sample_rate=%d, shape=%s", sr, y.shape)

        # ── Duration validation ───────────────────────────────────────────────
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info("Audio duration: %.2fs", duration)

        if duration < MIN_DURATION:
            raise ValueError(
                f"Audio too short. Minimum duration is {MIN_DURATION} seconds, "
                f"got {duration:.2f}s"
            )
        if duration > MAX_DURATION:
            raise ValueError(
                f"Audio too long. Maximum duration is {MAX_DURATION} seconds, "
                f"got {duration:.2f}s"
            )

        # ── Pre-normalization silence check ───────────────────────────────────
        # Catches truly flat / all-zero audio after codec decode.
        # Thresholds are very low because browser .webm speech can have
        # raw RMS as low as ~0.00003 due to MediaRecorder compression.
        logger.info("Checking for silence (pre-normalization)...")
        max_amp      = np.max(np.abs(y))
        raw_rms      = librosa.feature.rms(y=y)
        raw_mean_rms = float(np.mean(raw_rms))

        logger.info(
            "Audio stats: max_amp=%.8f, raw_mean_rms=%.8f, peak_rms_frame=%.8f",
            max_amp, raw_mean_rms, float(np.max(raw_rms)),
        )

        if max_amp < 1e-8:
            raise ValueError("Audio is silence. Please speak clearly.")

        if raw_mean_rms < 1e-6:
            raise ValueError("Audio is mostly silence. Please speak clearly.")

        # ── Normalize ─────────────────────────────────────────────────────────
        logger.info("Normalizing audio...")
        y = y / max_amp
        logger.info("Audio normalized (peak was %.8f)", max_amp)

        # ── Spectral flatness check (BEFORE noise reduction) ──────────────────
        # Run this on the raw normalized signal so codec noise hasn't been
        # removed yet — that's exactly what we want to detect here.
        # Silent/noise-only recordings are spectrally flat (flatness ≈ 0.8–1.0).
        # Real speech has structured formants (flatness ≈ 0.05–0.25).
        logger.info("Checking spectral flatness...")
        mean_flatness, voiced_ratio = _check_spectral_flatness(y, sr)

        if mean_flatness > FLATNESS_MAX_MEAN:
            raise ValueError(
                f"No speech detected — audio appears to be silence or background noise "
                f"(spectral flatness {mean_flatness:.2f}, expected < {FLATNESS_MAX_MEAN}). "
                f"Please speak clearly into your microphone."
            )

        if voiced_ratio < FLATNESS_MIN_VOICED:
            raise ValueError(
                f"No voiced speech detected (only {voiced_ratio * 100:.1f}% voiced frames). "
                f"Please speak clearly into your microphone."
            )

        # ── Voice Activity Detection ──────────────────────────────────────────
        # Secondary guard: checks that enough frames have above-threshold energy.
        logger.info("Running voice activity detection...")
        speech_ratio, mean_active_db = _detect_voice_activity(y, sr)

        if speech_ratio < VAD_MIN_SPEECH_RATIO:
            raise ValueError(
                f"No speech detected (only {speech_ratio * 100:.1f}% active frames). "
                f"Please speak clearly into your microphone."
            )

        if mean_active_db < VAD_MIN_ACTIVE_DB:
            raise ValueError(
                f"Audio too quiet to analyse (mean level {mean_active_db:.1f}dB). "
                f"Please move closer to your microphone."
            )

        logger.info(
            "VAD passed: %.1f%% speech frames, %.1fdB mean active level",
            speech_ratio * 100, mean_active_db,
        )

        # ── Noise Reduction ───────────────────────────────────────────────────
        logger.info("Applying noise reduction...")
        y = nr.reduce_noise(y=y, sr=sr, stationary=True)

        # ── Feature Extraction ────────────────────────────────────────────────
        logger.info("Extracting features...")
        features = extract_voice_features(y, sr)

        logger.info(
            "Audio processed successfully: duration=%.2fs, shape=%s",
            duration, y.shape,
        )
        return features, duration

    except ValueError as ve:
        logger.error("Validation error: %s", str(ve))
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        error_msg = str(e) if str(e) else repr(e)
        logger.error("Error processing audio: %s", error_msg)
        logger.error("Exception type: %s", type(e).__name__)
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to process audio file: {error_msg}")
    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.info("Temporary file deleted: %s", temp_file.name)
            except Exception as e:
                logger.warning("Failed to delete temporary file: %s", e)


def extract_voice_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:

    features = {}

    try:
        # 1. MFCC — 40 coefficients for depression/stress models
        logger.debug("Extracting MFCC features...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features["mfcc_mean"] = np.mean(mfcc, axis=1)
        features["mfcc_std"]  = np.std(mfcc,  axis=1)

        # 2. Pitch (F0)
        logger.debug("Extracting pitch features...")
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features["pitch_mean"] = float(np.mean(pitch_values))
                features["pitch_std"]  = float(np.std(pitch_values))
            else:
                features["pitch_mean"] = 0.0
                features["pitch_std"]  = 0.0
        except Exception as e:
            logger.warning("Failed to extract pitch: %s", e)
            features["pitch_mean"] = 0.0
            features["pitch_std"]  = 0.0

        # 3. Energy / RMS
        logger.debug("Extracting energy features...")
        rms = librosa.feature.rms(y=y)
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"]  = float(np.std(rms))

        # 4. Zero Crossing Rate
        logger.debug("Extracting zero crossing rate...")
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"]  = float(np.std(zcr))

        # 5. Spectral features
        logger.debug("Extracting spectral features...")
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"]  = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

        logger.debug(
            "Extracted features with %d MFCCs: %s", N_MFCC, list(features.keys())
        )
        return features

    except Exception as e:
        logger.error("Error extracting features: %s", str(e))
        logger.error(traceback.format_exc())
        raise