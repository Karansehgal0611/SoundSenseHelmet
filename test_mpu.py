from raspberry_pi.sensors.mpu6050 import HeadTracker

tracker = HeadTracker()

try:
    print("Starting MPU6050 test - move the sensor around")
    tracker.continuous_monitoring()
except Exception as e:
    print(f"Error: {e}")