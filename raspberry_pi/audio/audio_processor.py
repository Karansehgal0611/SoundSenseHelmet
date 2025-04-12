# audio_processor.py
import numpy as np
from features import extract_spectrogram

class AudioProcessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.spectrogram_queue = Queue(maxsize=10)
        
    def process_chunk(self, audio_chunk):
        """Convert raw audio to ML-ready features"""
        # Convert to mono if needed
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
            
        spectrogram = extract_spectrogram(
            audio_chunk,
            sample_rate=self.sample_rate,
            img_size=(64, 64)
        )
        return np.expand_dims(spectrogram, axis=0)  # Add batch dimension