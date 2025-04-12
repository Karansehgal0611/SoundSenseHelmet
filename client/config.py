"""
Configuration settings for the audio client.
"""

# Default Raspberry Pi connection settings
DEFAULT_HOST = "192.168.58.114" # Change to your Pi's IP address
DEFAULT_PORT = 12345

# Audio settings
AUDIO_FORMAT = "int16"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024  # Number of frames per buffer