# main.py
import time
from raspberry_pi.audio.audio_server import AudioServer

if __name__ == "__main__":
    server = AudioServer()
    server.start()
    
    try:
        print("Server is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()




'''# main.py (orchestrates the full flow)
from audio_server import AudioStream
from inference import RealTimeInference
#from output import HapticFeedback, VisualFeedback
import time

def main():
    # Initialize components
    audio_stream = AudioStream(sample_rate=22050, chunk_size=1103)  # 50ms chunks
    detector = RealTimeInference("models/emergency_sound_classifier.tflite")
    haptic = HapticFeedback()
    visual = VisualFeedback()
    
    # Start audio stream
    audio_stream.start_stream()
    
    try:
        print("Starting real-time detection...")
        while True:
            # [Laptop Microphone] → [Audio Stream]
            chunk = audio_stream.get_chunk()
            
            # [Audio Stream] → [Audio Processor]
            # [Audio Processor] → [Features]
            # [Features] → [Inference] → [Model]
            is_emergency, confidence = detector.process_frame(chunk)
            
            # [Model Output] → [Haptic/Visual]
            if is_emergency:
                print(f"EMERGENCY DETECTED! Confidence: {confidence:.2f}")
                #haptic.trigger()
                #visual.trigger()
            else:
                visual.idle()  # Optional: visual feedback for normal operation
                
            time.sleep(0.01)  # Small sleep to prevent CPU overload
            
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        audio_stream.stream.stop()
        audio_stream.stream.close()

if __name__ == "__main__":
    main()

'''




'''# main.py (runs on Pi)
from audio_server import AudioServer
from inference import RealTimeInference
#from output import HapticFeedback, VisualFeedback
import threading

def main():
    # Initialize components
    audio_server = AudioServer(port=5000)
    detector = RealTimeInference("ml/models/emergency_sound.tflite")
    haptic = HapticFeedback()
    visual = VisualFeedback()

    # Start audio server in separate thread
    server_thread = threading.Thread(
        target=audio_server.start_server,
        daemon=True
    )
    server_thread.start()

    try:
        print("Ready for audio stream...")
        while True:
            # Get audio chunk from network
            chunk = audio_server.get_chunk()
            
            # Process through ML pipeline
            is_emergency, confidence = detector.process_frame(chunk)
            
            # Trigger outputs
            if is_emergency:
                print(f"ALERT! Confidence: {confidence:.2f}")
                #haptic.trigger()
                #visual.trigger()
                
    except KeyboardInterrupt:
        audio_server.stop_server()
        print("System stopped")

if __name__ == "__main__":
    main()
'''