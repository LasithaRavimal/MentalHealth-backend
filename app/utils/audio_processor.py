import librosa
import numpy as np
from typing import Tuple, Dict, Optional
import logging
import tempfile
import os
import traceback
import noisereduce as nr

logger = logging.getLogger(__name__)

# Audio constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_DURATION = 3  # seconds
MAX_DURATION = 120  # seconds
SAMPLE_RATE = 22050  # Standard for speech analysis
N_MFCC = 40  # Changed to 40 for depression,stress models
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")


def validate_audio_file(audio_bytes: bytes, filename: str) -> Optional[str]:
    
    # Check file size
    if len(audio_bytes) > MAX_FILE_SIZE:
        return f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
    
    # Check minimum file size
    if len(audio_bytes) < 100:
        return "Audio file appears to be empty"
    
    # Check file extension
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return None


def process_audio_file(audio_bytes: bytes) -> Tuple[Dict[str, np.ndarray], float]:
   
    temp_file = None
    try:
        logger.info(f"Starting audio processing, file size: {len(audio_bytes)} bytes")
        
        #saving the audio to a temporary file on the disk so Librosa can access it via a standard file path.
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ogg')
        temp_file.write(audio_bytes)
        temp_file.close()
        
        logger.info(f"Temporary file created: {temp_file.name}")
        
        # Load audio from temporary file
        logger.info("Loading audio with librosa...")
        y, sr = librosa.load(temp_file.name, sr=SAMPLE_RATE, mono=True)
        logger.info(f"Audio loaded: sample_rate={sr}, shape={y.shape}")
        
        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Audio duration: {duration:.2f}s")
        
        # Validate duration
        if duration < MIN_DURATION:
            raise ValueError(f"Audio too short. Minimum duration is {MIN_DURATION} seconds, got {duration:.2f}s")
        if duration > MAX_DURATION:
            raise ValueError(f"Audio too long. Maximum duration is {MAX_DURATION} seconds, got {duration:.2f}s")
        
        # Silence Detection (before normalization, using raw audio levels)
        logger.info("Checking for silence...")
        max_amp = np.max(np.abs(y))
        logger.info(f"Audio peak amplitude: {max_amp:.6f}")
        
        if max_amp < 1e-6:
            raise ValueError("Audio is silence. Please speak clearly.")
        
        raw_rms = librosa.feature.rms(y=y)
        raw_mean_rms = float(np.mean(raw_rms))
        logger.info(f"Audio mean RMS energy (raw): {raw_mean_rms:.6f}")
        
        if raw_mean_rms < 0.0005:
            raise ValueError("Audio is mostly silence. Please speak clearly.")
        
        # Normalize audio
        logger.info("Normalizing audio...")
        y = y / max_amp
        logger.info(f"Audio normalized (peak was {max_amp:.6f})")
            
        #Noise Reduction
        logger.info("Applying noise reduction...")
        y = nr.reduce_noise(y=y, sr=sr, stationary=True)
        
        # Extract features
        logger.info("Extracting features...")
        features = extract_voice_features(y, sr)
        
        logger.info(f"Audio processed successfully: duration={duration:.2f}s, shape={y.shape}")
        return features, duration
        
    except ValueError as ve:
        # Re-raise ValueError with original message
        logger.error(f"Validation error: {str(ve)}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        error_msg = str(e) if str(e) else repr(e)
        logger.error(f"Error processing audio: {error_msg}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to process audio file: {error_msg}")
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.info(f"Temporary file deleted: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


def extract_voice_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    
    features = {}
    
    try:
        # 1. MFCC (40 coefficients for depression model)
        logger.debug("Extracting MFCC features...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # 2. Pitch (F0) - Voice fundamental frequency
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
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
        except Exception as e:
            logger.warning(f"Failed to extract pitch: {e}")
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
        
        # 3. Energy/RMS
        logger.debug("Extracting energy features...")
        rms = librosa.feature.rms(y=y)
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        
        # 4. Zero Crossing Rate
        logger.debug("Extracting zero crossing rate...")
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # 5. Spectral features
        logger.debug("Extracting spectral features...")
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        logger.debug(f"Extracted features with {N_MFCC} MFCCs: {list(features.keys())}")
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.error(traceback.format_exc())
        raise
