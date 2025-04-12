import sounddevice as sd
import socket
import numpy as np
from queue import Queue
import argparse

class AudioClient:
    def __init__(self, pi_ip, port=5000, sample_rate=22050, chunk_size=2048):
        """
        Initialize audio client for streaming from laptop mic to Pi
        
        Args:
            pi_ip: Raspberry Pi IP address
            port: UDP port to use
            sample_rate: Audio sample rate
            chunk_size: Samples per chunk (~50ms at 22050Hz)
        """
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = (pi_ip, port)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = Queue(maxsize=10)
        self.stream = None

    def callback(self, indata, frames, time, status):
        """Called for each audio chunk from laptop mic"""
        if status:
            print(f"Audio status: {status}")
        print(f"Sending {len(indata)} samples")
        self.audio_queue.put(indata.copy())

    def start_stream(self):
        """Start streaming to Raspberry Pi"""
        print(f"Starting stream to {self.server_address}...")
        
        # Audio stream configuration
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.callback,
            blocksize=self.chunk_size
        )
        
        try:
            self.stream.start()
            print(f"Streaming audio (SR: {self.sample_rate}, Chunk: {self.chunk_size})")
            
            while True:
                chunk = self.audio_queue.get()
                self.udp_socket.sendto(chunk.tobytes(), self.server_address)
                
        except KeyboardInterrupt:
            print("\nStopping stream...")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.udp_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pi-ip', required=True, help='Raspberry Pi IP address')
    parser.add_argument('--port', type=int, default=5000, help='UDP port')
    parser.add_argument('--rate', type=int, default=22050, help='Sample rate')
    parser.add_argument('--chunk', type=int, default=1103, help='Chunk size')
    args = parser.parse_args()

    client = AudioClient(
        pi_ip=args.pi_ip,
        port=args.port,
        sample_rate=args.rate,
        chunk_size=args.chunk
    )
    client.start_stream()


