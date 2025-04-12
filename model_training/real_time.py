import numpy as np
import pandas as pd
import librosa
import cv2
import os
import time
import threading
import queue
import tensorflow as tf
import pyaudio
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output
from tensorflow.keras.models import load_model

class RealTimeAudioClassifier:
    def __init__(self, model_path, window_size=1.0, hop_size=0.5, sample_rate=22050, img_size=(64, 64)):
        """
        Initialize the real-time audio classifier.
        
        Args:
            model_path (str): Path to the trained model file (.h5)
            window_size (float): Size of audio window in seconds
            hop_size (float): Hop size between windows in seconds
            sample_rate (int): Audio sample rate
            img_size (tuple): Size of spectrogram image (height, width)
        """
        self.model = load_model(model_path)
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.img_size = img_size
        
        # Calculate buffer sizes
        self.window_samples = int(window_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)
        
        # Audio buffer for overlap-add processing
        self.audio_buffer = np.zeros(self.window_samples)
        
        # Queue for results
        self.results_queue = queue.Queue()
        
        # Flags
        self.is_running = False
        self.emergency_detected = False
        self.emergency_count = 0
        self.classification_history = []
        
        # Classification threshold
        self.threshold = 0.7
        
        # Classification classes
        self.classes = {0: "Normal Sound", 1: "Emergency Sound"}
    
    def extract_features(self, audio):
        """
        Extract spectrogram features from audio segment.
        
        Args:
            audio (np.array): Audio time series
            
        Returns:
            np.array: Processed spectrogram ready for model input
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
        
        Args:
            indata (np.array): Input audio data
            frames (int): Number of frames
            time_info (dict): Time information
            status (int): Status flag
        """
        if not self.is_running:
            return
        
        # Get the audio data and convert to float32
        audio_chunk = indata[:, 0].astype(np.float32)
        
        # Update buffer
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_chunk))
        self.audio_buffer[-len(audio_chunk):] = audio_chunk
        
        # Process audio
        threading.Thread(target=self.process_audio, args=(self.audio_buffer.copy(),)).start()
    
    def process_audio(self, audio):
        """
        Process audio segment and make prediction.
        
        Args:
            audio (np.array): Audio time series
        """
        features = self.extract_features(audio)
        
        if features is not None:
            # Make prediction
            prediction = self.model.predict(features, verbose=0)[0][0]
            
            # Get class and confidence
            predicted_class = 1 if prediction >= self.threshold else 0
            confidence = prediction if predicted_class == 1 else 1 - prediction
            
            # Update emergency detection
            if predicted_class == 1:
                self.emergency_count += 1
                if self.emergency_count >= 3:  # Require 3 consecutive detections
                    self.emergency_detected = True
            else:
                self.emergency_count = max(0, self.emergency_count - 1)
                if self.emergency_count == 0:
                    self.emergency_detected = False
            
            # Store result
            result = {
                'class': predicted_class,
                'class_name': self.classes[predicted_class],
                'confidence': confidence,
                'emergency_detected': self.emergency_detected,
                'timestamp': time.time()
            }
            
            self.classification_history.append(result)
            if len(self.classification_history) > 100:
                self.classification_history.pop(0)
                
            self.results_queue.put(result)
    
    def start_streaming(self):
        """Start the audio stream."""
        self.is_running = True
        
        # Configure audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.hop_samples
        )
        
        # Start the stream
        self.stream.start()
        print("Real-time audio classification started!")
    
    def stop_streaming(self):
        """Stop the audio stream."""
        if hasattr(self, 'stream'):
            self.is_running = False
            self.stream.stop()
            self.stream.close()
            print("Real-time audio classification stopped!")
    
    def get_latest_result(self):
        """Get the latest classification result."""
        if not self.results_queue.empty():
            return self.results_queue.get()
        return None

    def visualize_real_time(self):
        """
        Create a real-time visualization of classification results.
        """
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plt.tight_layout()
        
        # For the spectrogram display
        img_display = ax1.imshow(np.zeros(self.img_size), 
                               aspect='auto', 
                               origin='lower', 
                               cmap='viridis',
                               vmin=-80, vmax=0)  # Set reasonable dB limits
        ax1.set_title('Real-time Spectrogram')
        ax1.set_xlabel('Time')
        ax1. set_ylabel('Frequency')
        
        # For the classification results
        max_points = 50  # Number of points to display
        confidence_data = np.zeros(max_points)
        timestamps = np.zeros(max_points)
        line, = ax2.plot(timestamps, confidence_data, 'r-')
        ax2.set_ylim(0, 1)
        ax2.set_title('Classification Confidence')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Confidence')
        ax2.axhline(y=self.threshold, color='g', linestyle='--', label=f'Threshold ({self.threshold})')
        ax2.legend()
        
        text = ax2.text(0.02, 0.95, 'Starting...', transform=ax2.transAxes, 
                       fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        try:
            while self.is_running:
                result = self.get_latest_result()
                
                if result is not None:
                    # Update confidence plot
                    confidence_data = np.roll(confidence_data, -1)
                    timestamps = np.roll(timestamps, -1)
                    
                    current_time = time.time()
                    if timestamps[-2] == 0:  # First data point
                        timestamps[-1] = 0
                    else:
                        timestamps[-1] = current_time - timestamps[0]
                    
                    confidence_data[-1] = result['confidence'] if result['class'] == 1 else 1 - result['confidence']
                    
                    line.set_data(np.arange(max_points), confidence_data)
                    ax2.set_xlim(0, max_points)
                    
                    # Update status text
                    status = (f"Detected: {result['class_name']}\n"
                              f"Confidence: {result['confidence']:.2f}\n"
                              f"Emergency: {'YES' if result['emergency_detected'] else 'NO'}")
                    text.set_text(status)
                    
                    # Set text color based on detection
                    text.set_color('red' if result['emergency_detected'] else 'black')
                    
                    # Update spectrogram display if we have audio data
                    if hasattr(self, 'audio_buffer'):
                        try:
                            spect = librosa.feature.melspectrogram(y=self.audio_buffer, 
                                                                sr=self.sample_rate,
                                                                n_fft=min(2048, len(self.audio_buffer)))
                            spect = librosa.power_to_db(spect, ref=np.max)
                            spect = cv2.resize(spect, self.img_size)
                            img_display.set_array(spect)
                            img_display.set_clim(vmin=spect.min(), vmax=spect.max())
                        except Exception as e:
                            print(f"Spectrogram error: {e}")
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.05)  # Small delay to prevent 100% CPU usage
                
        except KeyboardInterrupt:
            print("Visualization stopped by user")
        finally:
            plt.ioff()



def run_demo(model_path):
    """
    Run a demo of the real-time audio classifier.
    
    Args:
        model_path (str): Path to the trained model file (.h5)
    """
    # Initialize classifier
    classifier = RealTimeAudioClassifier(model_path)
    
    try:
        # Start streaming
        classifier.start_streaming()
        
        # Visualize results in real-time
        classifier.visualize_real_time()
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop streaming
        classifier.stop_streaming()


if __name__ == "__main__":
    # Path to your trained model
    model_path = "/home/karan/JupyterKaran/SoundClassifier/emergency_sound_model.h5"
    
    # Run demo
    run_demo(model_path)