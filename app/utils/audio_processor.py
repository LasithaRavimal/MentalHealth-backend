import librosa
import numpy as np
import soundfile as sf
import io
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Audio constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_DURATION = 3  # seconds (reduced for depression model)
MAX_DURATION = 120  # seconds
SAMPLE_RATE = 22050  # Standard for speech analysis
N_MFCC = 40  # Changed to 40 for depression model


def validate_audio_file(audio_bytes: bytes, filename: str) -> Optional[str]:
    """
    Validate audio file size and format.
    Returns error message if invalid, None if valid.
    """
    # Check file size
    if len(audio_bytes) > MAX_FILE_SIZE:
        return f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
    
    # Check file extension
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        return f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}"
    
    return None


def process_audio_file(audio_bytes: bytes) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Process audio file and extract voice features.
    
    Args:
        audio_bytes: Raw audio file bytes
        
    Returns:
        Tuple of (features_dict, duration_in_seconds)
        
    Raises:
        ValueError: If audio processing fails
    """
    try:
        # Load audio from bytes
        audio_io = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_io, sr=SAMPLE_RATE, mono=True)
        
        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Validate duration
        if duration < MIN_DURATION:
            raise ValueError(f"Audio too short. Minimum duration is {MIN_DURATION} seconds")
        if duration > MAX_DURATION:
            raise ValueError(f"Audio too long. Maximum duration is {MAX_DURATION} seconds")
        
        # Extract features
        features = extract_voice_features(y, sr)
        
        logger.info(f"Audio processed successfully: duration={duration:.2f}s, shape={y.shape}")
        return features, duration
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise ValueError(f"Failed to process audio file: {str(e)}")


def extract_voice_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    Extract MFCC and other voice features from audio signal.
    Now extracts 40 MFCC coefficients for depression model compatibility.
    
    Args:
        y: Audio time series
        sr: Sample rate
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # 1. MFCC (40 coefficients for depression model)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_std'] = np.std(mfcc, axis=1)
    
    # 2. Pitch (F0) - Voice fundamental frequency
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
    rms = librosa.feature.rms(y=y)
    features['energy_mean'] = float(np.mean(rms))
    features['energy_std'] = float(np.std(rms))
    
    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    
    # 5. Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
    
    logger.debug(f"Extracted features with {N_MFCC} MFCCs: {list(features.keys())}")
    return features