# SoundSenseHelmet 🛡️🎧  
**AI-Powered Safety Helmet for the Deaf Community**  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-2.10+-orange.svg)](https://www.tensorflow.org/lite)  

<div align="center">
  <img src="docs/helmet_demo.gif" width="400" alt="Demo">
</div>

## 🏗️ Project Structure

```markdown
smart_helmet/ ├── client/ # Laptop audio streaming client │ └── audio_client.py ├── raspberry_pi/ # Core helmet system │ ├── audio/ # Audio processing │ │ ├── audio_server.py # UDP server │ │ ├── audio_processor.py # Chunk handling │ │ └── features.py # Spectrogram conversion │ ├── ml/ # Machine learning │ │ ├── inference.py # Real-time prediction │ │ ├── model_loader.py # TFLite integration │ │ └── models/ # Pretrained models │ ├── sensors/ # Sensor interfaces │ │ ├── force_sensor.py # Impact detection │ │ ├── gps.py # Location tracking │ │ └── mpu6050.py # Head tracking │ ├── output/ # User feedback │ │ ├── haptic.py # Vibration control │ │ └── visual.py # LED patterns │ ├── utils/ # Utilities │ │ ├── alerts.py # Notification system │ │ ├── logger.py # Logging config │ │ └── config.py # Hardware pin config ├── model_training/ # ML model development ├── tests/ # Unit tests │ ├── test_audio_streaming.py │ ├── test_ml_inference.py │ └── test_sensors.py ├── requirements.txt # Python dependencies └── README.md
```

## 🚀 Key Features
- **Real-Time Audio Classification**  
  - Processes microphone input at 22.05kHz (50ms chunks)
  - Identifies sirens, horns, and emergency sounds
- **Multi-Sensor Integration**  
  - MPU6050 for head orientation tracking  
  - Force sensor for crash detection
  - GPS for emergency location logging
- **Tactile Feedback System**  
  - Configurable vibration patterns for different alerts
  - LED visual indicators

## 📦 Hardware  
- Raspberry Pi 4  
- MEMS Microphone  
- MPU6050 (Head tracking)  
- GPS Module (NEO-6M)  
- Vibration Motors  

## 🛠️ Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure hardware pins
nano raspberry_pi/config.py

# 3. Start the system
# On Pi:
python raspberry_pi/main.py

# On laptop:
python client/audio_client.py --pi-ip 192.168.x.x
```
## 🧪 Testing
```bash
# Run all tests
python -m pytest tests/

# Individual test modules
pytest tests/test_sensors.py
pytest tests/test_ml_inference.py -v
```

---

### **Recommended Tags**  
```markdown
Topics:  
assistive-technology, deaf-community, real-time-audio, raspberry-pi, tensorflow-lite, haptic-feedback, smart-helmet, accessibility
```
