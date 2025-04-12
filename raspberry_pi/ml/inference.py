# raspberry_pi/ml/inference.py
import numpy as np
import logging
from raspberry_pi.audio.features import extract_spectrogram  # Relative import
from .model_loader import ModelLoader     # Relative import

class RealTimeInference:
    def __init__(self, model_loader, threshold=0.70):
        self.model = model_loader
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Log expected input shape
        input_shape = self.model.input_details[0]['shape']
        self.logger.info(f"Model expects input shape: {input_shape}")

    def process_frame(self, audio_chunk):
        """Process audio chunk with shape verification"""
        try:
            # Extract features
            spectrogram = extract_spectrogram(audio_chunk)
            print(f"Spectrogram shape: {spectrogram.shape}")
            
            # Verify shape
            expected_shape = self.model.input_details[0]['shape']
            if spectrogram.shape != tuple(expected_shape):
                self.logger.error(
                    f"Shape mismatch! Got {spectrogram.shape}, "
                    f"expected {expected_shape}"
                )
                return False, 0.0
            
            # Run inference
            prediction = self.model.predict(spectrogram)
            print(f"Raw prediction: {prediction}")
            return prediction > self.threshold, prediction
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return False, 0.0














'''
# inference.py
from model_loader import ModelLoader
from audio_processor import AudioProcessor

class RealTimeInference:
    def __init__(self, model_path):
        self.model = ModelLoader(model_path)
        self.processor = AudioProcessor()
        self.threshold = 0.70  # Emergency detection threshold
        
    def process_frame(self, audio_chunk):
        """Complete processing pipeline for one audio chunk"""
        features = self.processor.process_chunk(audio_chunk)
        prediction = self.model.predict(features)
        return prediction > self.threshold, prediction


import numpy as np
import time
import threading
import queue
import logging
from .model_loader import load_model, run_inference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InferenceEngine")

class InferenceEngine:
    def __init__(self, model_path, feature_extractor, config):
        """
        Initialize the inference engine for emergency sound detection.
        
        Args:
            model_path (str): Path to the TFLite model
            feature_extractor (callable): Function to extract features from raw audio
            config (dict): Configuration parameters including:
                - threshold: Detection threshold (float between 0-1)
                - window_size: Size of audio window to process (samples)
                - sample_rate: Audio sample rate (Hz)
                - overlap: Overlap between consecutive windows (0-1)
        """
        self.config = config
        self.threshold = config.get('threshold', 0.7)
        self.window_size = config.get('window_size', 16000)  # 1 second at 16kHz
        self.sample_rate = config.get('sample_rate', 16000)
        self.overlap = config.get('overlap', 0.5)  # 50% overlap by default
        
        # Load TFLite model
        logger.info(f"Loading model from {model_path}")
        self.interpreter, self.input_details, self.output_details, self.input_shape = load_model(model_path)
        
        # Audio feature extraction function
        self.feature_extractor = feature_extractor
        
        # Setup processing queue and buffer
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.array([], dtype=np.int16)
        self.is_running = False
        self.processing_thread = None
        
        # Callbacks
        self.on_detection = None
        self.on_normal = None
        
        logger.info(f"Inference engine initialized with threshold: {self.threshold}")
        logger.info(f"Input shape expected: {self.input_shape}")

    def set_callbacks(self, on_detection=None, on_normal=None):
        """
        Set callback functions for detection events.
        
        Args:
            on_detection (callable): Called when emergency sound is detected
            on_normal (callable): Called when no emergency sound is detected
        """
        self.on_detection = on_detection
        self.on_normal = on_normal

    def add_audio_chunk(self, audio_chunk):
        """
        Add an audio chunk to the processing queue.
        
        Args:
            audio_chunk (bytes): Raw audio data
        """
        if self.is_running:
            self.audio_queue.put(audio_chunk)

    def _process_audio_loop(self):
        """Background thread for audio processing and inference."""
        last_detection_time = 0
        detection_cooldown = 1.0  # Seconds to wait before another detection alert
        
        while self.is_running:
            try:
                # Process any audio in the queue
                while not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    
                    # Add to buffer
                    self.audio_buffer = np.append(self.audio_buffer, audio_data)
                
                # Keep buffer at a reasonable size
                max_buffer_size = int(self.window_size * 2)  # Keep at most 2 windows
                if len(self.audio_buffer) > max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                
                # Only process if we have enough audio
                if len(self.audio_buffer) >= self.window_size:
                    # Extract window for processing
                    audio_window = self.audio_buffer[-self.window_size:]
                    
                    # Extract features
                    features = self.feature_extractor(audio_window, self.sample_rate)
                    
                    # Run inference
                    result = run_inference(self.interpreter, self.input_details, 
                                           self.output_details, features)
                    
                    current_time = time.time()
                    score = result[0][0] if result.shape[1] == 1 else result[0][1]
                    
                    # Check against threshold
                    if score > self.threshold:
                        logger.info(f"EMERGENCY SOUND DETECTED with confidence: {score:.4f}")
                        
                        # Only trigger detection if cooldown period has passed
                        if current_time - last_detection_time > detection_cooldown:
                            if self.on_detection:
                                self.on_detection(score)
                            last_detection_time = current_time
                    else:
                        if self.on_normal:
                            self.on_normal(score)
                
                # Move window for next iteration (with overlap)
                step_size = int(self.window_size * (1 - self.overlap))
                if len(self.audio_buffer) > step_size:
                    self.audio_buffer = self.audio_buffer[step_size:]
                
                # Sleep a tiny bit to avoid hogging CPU
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop if there's an error
        
        logger.info("Processing loop stopped")

    def start(self):
        """Start the inference engine."""
        if not self.is_running:
            logger.info("Starting inference engine")
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._process_audio_loop,
                daemon=True
            )
            self.processing_thread.start()
            return True
        return False

    def stop(self):
        """Stop the inference engine."""
        if self.is_running:
            logger.info("Stopping inference engine")
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            self.audio_buffer = np.array([], dtype=np.int16)
            while not self.audio_queue.empty():
                self.audio_queue.get()
            return True
        return False
    
    def is_active(self):
        """Check if the inference engine is running."""
        return self.is_running


# Example usage (can be removed in production)
if __name__ == "__main__":
    # Example feature extractor (replace with your actual implementation)
    def dummy_feature_extractor(audio, sample_rate):
        """Dummy function to demonstrate feature extraction signature."""
        # In a real implementation, this would extract MFCCs, mel spectrograms, etc.
        return np.random.random((128, 128))
    
    # Example config
    config = {
        'threshold': 0.7,
        'window_size': 16000,
        'sample_rate': 16000,
        'overlap': 0.5
    }
    
    # Example callbacks
    def on_detection_callback(score):
        print(f"ALERT! Emergency sound detected with score: {score}")
    
    def on_normal_callback(score):
        print(f"Normal sound, score: {score}")
    
    # Create and start engine
    engine = InferenceEngine(
        model_path="../models/emergency_sound_classifier.tflite",
        feature_extractor=dummy_feature_extractor,
        config=config
    )
    
    engine.set_callbacks(
        on_detection=on_detection_callback,
        on_normal=on_normal_callback
    )
    
    engine.start()
    
    # Simulate adding audio chunks
    try:
        for _ in range(10):
            # Generate random audio data
            fake_audio = np.random.randint(-32768, 32767, 1600, dtype=np.int16).tobytes()
            engine.add_audio_chunk(fake_audio)
            time.sleep(0.1)
    finally:
        engine.stop()
'''