import os
import sys
import socket
import numpy as np
from queue import Queue
import threading
import logging
import time

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from raspberry_pi.ml.model_loader import ModelLoader
from raspberry_pi.ml.inference import RealTimeInference
from raspberry_pi.output.haptic import HapticController
#from raspberry_pi.sensors.mpu6050 import HeadTracker
from raspberry_pi.output.visual import LEDController

class AudioServer:
    def __init__(self, port=5000, model_path="../ml/models/emergency_sound_classifier.tflite"):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('0.0.0.0', port))
        self.audio_queue = Queue(maxsize=20)
        self.running = False
        
        # Initialize ML components
        # Get absolute path to model
        model_path = "ml/models/emergency_sound_classifier.tflite"

        self.model = ModelLoader(model_path)
        self.inference = RealTimeInference(self.model)
        self.vibrator = HapticController(motor_pin=12)  
        #self.head_tracker = HeadTracker(led_pin=27)
        self.led = LEDController(17)

        # Thread for head tracking
        '''self.tracking_thread = threading.Thread(
            target=self._track_head_loop,
            daemon=True
        )'''

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _handle_emergency(self, confidence):
        """Centralized emergency response"""
        print(f"EMERGENCY! Confidence: {confidence:.2f}")
        self.led.blink(times=5, speed=0.2)
        self.vibrator.emergency_sos()  # <-- Trigger vibration pattern
        '''if self.head_tracker.check_head_tilt():
            self.led.police_siren_pattern()''' 


    def _receive_loop(self):
        """Continuous audio reception and processing"""
        while self.running:
            try:
                # Receive audio chunk
                data, _ = self.udp_socket.recvfrom(4096)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                print(f"Received audio chunk: {len(audio_chunk)} samples")  # Debug

                
                # Process through ML pipeline
                is_emergency, confidence = self.inference.process_frame(audio_chunk)
                print(f"Processed - Emergency: {is_emergency}, Confidence: {confidence:.2f}")
                
                # Trigger outputs
                if is_emergency:
                    self._handle_emergency(confidence)
                    
                    
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
    
    '''def _track_head_loop(self):
        """Continuous head position monitoring"""
        while self.running:
            try:
                # Head tracking runs independently
                if self.head_tracker.check_head_tilt():
                    self.logger.warning("Abnormal head position detected!")
                    # LED controlled automatically by HeadTracker class
                    
                time.sleep(0.1)  # Reduce CPU usage
                
            except Exception as e:
                self.logger.error(f"Head tracking error: {e}")'''

    def start(self):
        """Start the audio server"""
        if not self.running:
            self.running = True
            self.receiver_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True
            )
            # Start audio processing
            self.receiver_thread.start()
            self.logger.info(f"Audio server started on port 5000")

            # Start head tracking
            #self.tracking_thread.start()
            #self.logger.info("System started - audio and head tracking active")

    def stop(self):
        """Stop the audio server"""
        self.running = False
        self.udp_socket.close()
        #self.haptic.cleanup()
        self.led.cleanup()
        #self.head_tracker.led.cleanup()
        self.logger.info("Audio server stopped")

if __name__ == "__main__":
    server = AudioServer()
    try:
        server.start()
        while True:  # Keep main thread alive
            pass
    except KeyboardInterrupt:
        server.stop()


'''import socket
import numpy as np
from queue import Queue
import threading
import argparse

class AudioServer:
    def __init__(self, port=5000, sample_rate=22050, chunk_size=1103):
        """
        Initialize audio server to receive streams from client
        
        Args:
            port: UDP port to listen on
            sample_rate: Expected audio sample rate
            chunk_size: Samples per audio chunk
        """
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('0.0.0.0', port))
        self.audio_queue = Queue(maxsize=20)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.port = port

    def _receive_loop(self):
        """Continuous reception of audio data"""
        while self.running:
            try:
                data, _ = self.udp_socket.recvfrom(4096)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_chunk)
            except Exception as e:
                if self.running:  # Only log if not during shutdown
                    print(f"Receive error: {e}")

    def start_server(self):
        """Start receiving audio stream in background thread"""
        if not self.running:
            self.running = True
            self.receiver_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True
            )
            self.receiver_thread.start()
            print(f"Server listening on port {self.port} (SR: {self.sample_rate})")

    def get_chunk(self):
        """Get next audio chunk (blocking)"""
        return self.audio_queue.get()

    def stop_server(self):
        """Cleanup network resources"""
        self.running = False
        self.udp_socket.close()
        if hasattr(self, 'receiver_thread'):
            self.receiver_thread.join(timeout=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='UDP port')
    parser.add_argument('--rate', type=int, default=22050, help='Sample rate')
    parser.add_argument('--chunk', type=int, default=1103, help='Chunk size')
    args = parser.parse_args()

    server = AudioServer(
        port=args.port,
        sample_rate=args.rate,
        chunk_size=args.chunk
    )
    
    try:
        server.start_server()
        while True:
            chunk = server.get_chunk()
            print(f"Received chunk: {len(chunk)} samples")
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.stop_server()
'''



