import numpy as np
import librosa
import cv2
import tensorflow as tf
import sounddevice as sd
import time
import threading
import queue
from tensorflow.keras.models import load_model

class SimpleAudioClassifier:
    def __init__(self, model_path, window_size=1.0, sample_rate=22050, img_size=(64, 64)):
        """
        Simple real-time audio classifier for emergency sound detection.
        
        Args:
            model_path (str): Path to the trained model file (.h5)
            window_size (float): Size of audio window in seconds
            sample_rate (int): Audio sample rate
            img_size (tuple): Size of spectrogram image (height, width)
        """
        print("Loading model...")
        self.model = load_model(model_path)
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.img_size = img_size
        
        # Calculate buffer size
        self.window_samples = int(window_size * sample_rate)
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.window_samples)
        
        # Flags
        self.is_running = False
        
        # Threshold for detection
        self.threshold = 0.7
        
        print("Classifier initialized")
    
    def extract_features(self, audio):
        """
        Extract spectrogram features from audio segment.
        """
        try:
            # Create spectrogram
            spect = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
            spect = librosa.power_to_db(spect, ref=np.max)
            
            # Resize spectrogram
            spect = cv2.resize(spect, self.img_size)
            
            # Normalize
            spect = (spect - spect.min()) / (spect.max() - spect.min() + 1e-8)
            
            # Reshape for model input (batch_size, height, width, channels)
            spect = spect.reshape(1, self.img_size[0], self.img_size[1], 1)
            
            return spect
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for audio stream.
        """
        if status:
            print(f"Status: {status}")
        
        # Get the audio data and convert to float32
        audio_chunk = indata[:, 0].astype(np.float32)
        
        # Update buffer with new audio data
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_chunk))
        self.audio_buffer[-len(audio_chunk):] = audio_chunk
        
        # Process in a separate thread to prevent audio dropouts
        threading.Thread(target=self.process_audio).start()
    
    def process_audio(self):
        """
        Process current audio buffer and make prediction.
        """
        features = self.extract_features(self.audio_buffer)
        
        if features is not None:
            # Make prediction
            prediction = self.model.predict(features, verbose=0)[0][0]
            
            # Get class
            is_emergency = prediction >= self.threshold
            
            # Display result
            if is_emergency:
                print(f"\033[91m[EMERGENCY DETECTED] Confidence: {prediction:.2f}\033[0m")
            else:
                print(f"Normal sound. Confidence: {1-prediction:.2f}")
    
    def start(self):
        """Start the audio classification."""
        self.is_running = True
        
        print("Starting audio stream...")
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=int(0.2 * self.sample_rate)  # Process in 200ms chunks
        )
        
        self.stream.start()
        print("Real-time classification started! Press Ctrl+C to stop.")
    
    def stop(self):
        """Stop the audio classification."""
        if hasattr(self, 'stream'):
            self.is_running = False
            self.stream.stop()
            self.stream.close()
            print("Classification stopped.")


def main():
    """Main function to run the simple classifier."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time audio emergency sound detection')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model (.h5)')
    parser.add_argument('--threshold', type=float, default=0.7, help='Detection threshold (0-1)')
    args = parser.parse_args()
    
    # Create classifier
    classifier = SimpleAudioClassifier(args.model)
    classifier.threshold = args.threshold
    
    try:
        # Start classification
        classifier.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        classifier.stop()


if __name__ == "__main__":
    main()