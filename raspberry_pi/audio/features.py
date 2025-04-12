# raspberry_pi/ml/features.py
import numpy as np
import librosa
import cv2

def extract_spectrogram(audio_chunk, sample_rate=22050, img_size=(64, 64)):
    """Convert audio chunk to properly shaped spectrogram"""
    # Ensure audio is mono
    if len(audio_chunk.shape) > 1:
        audio_chunk = np.mean(audio_chunk, axis=0)
    
    n_fft = min(1024, len(audio_chunk))  # Dynamic FFT size
    hop_length = n_fft // 4  # 25% overlap

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=img_size[0],
        center = False
    )
    log_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize and resize
    norm_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min() + 1e-7)
    resized = cv2.resize(norm_spec, img_size)
    
    # Add batch and channel dimensions (now shape will be [1,64,64,1])
    return np.expand_dims(np.expand_dims(resized, 0), -1)


'''# features.py (optimized version)
import librosa
import numpy as np
import cv2

def extract_spectrogram(audio_chunk, sample_rate=22050, img_size=(64, 64)):
    """Optimized spectrogram extraction for real-time"""
    # Use pre-computed mel basis for faster processing
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=2048,
        n_mels=img_size[0]
    )
    
    # STFT with pre-allocation
    stft = librosa.stft(
        audio_chunk,
        n_fft=2048,
        hop_length=512,
        win_length=1024,
        center=False
    )
    
    # Mel spectrogram
    spect = np.dot(mel_basis, np.abs(stft)**2)
    spect = librosa.power_to_db(spect, ref=np.max)
    
    # Optimized resize and normalize
    spect = cv2.resize(spect, img_size, interpolation=cv2.INTER_AREA)
    spect = (spect - np.min(spect)) / (np.max(spect) - np.min(spect) + 1e-7)
    return spect.astype(np.float32)
'''
'''import librosa
import numpy as np

def extract_features(audio, sample_rate=16000):
    """Convert raw audio to MFCC features"""
    # Normalize audio
    audio = audio.astype(np.float32) / 32768.0
    
    # Extract MFCCs (adjust parameters to match your model's training)
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40,
        n_fft=512,
        hop_length=160
    )
    
    # Normalize MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    return mfccs.T  # Transpose to (time, features)
'''