from mpu6050 import mpu6050
from raspberry_pi.output.visual import LEDController
import time
import math
from collections import deque  # For smoothing

class HeadTracker:
    def __init__(self, led_pin=27):
        self.sensor = mpu6050(0x68)
        self.led = LEDController(led_pin)  # Dedicated LED
        self.angle_buffer = deque(maxlen=5)  # For smoothing
        self.calibrate()

    def calibrate(self):
        """Run while keeping the sensor flat"""
        print("Calibrating - keep sensor flat for 2 seconds...")
        offsets = {'x': 0, 'y': 0, 'z': 0}
        samples = 100
        
        for _ in range(samples):
            accel = self.sensor.get_accel_data()
            offsets['x'] += accel['x']
            offsets['y'] += accel['y']
            offsets['z'] += accel['z']
            time.sleep(0.02)
        
        self.offset = {k: v/samples for k,v in offsets.items()}
        print(f"Calibration offsets: {self.offset}")

    def get_orientation(self):
        """Returns smoothed pitch/roll in degrees"""
        accel = self.sensor.get_accel_data()
        # Apply calibration
        accel['x'] -= self.offset['x']
        accel['y'] -= self.offset['y']
        accel['z'] -= self.offset['z']
        
        pitch = math.degrees(math.atan2(accel['y'], accel['z']))
        roll = math.degrees(math.atan2(-accel['x'], math.sqrt(accel['y']**2 + accel['z']**2)))
        
        # Smooth readings
        self.angle_buffer.append((pitch, roll))
        avg_pitch = sum(p for p,r in self.angle_buffer)/len(self.angle_buffer)
        avg_roll = sum(r for p,r in self.angle_buffer)/len(self.angle_buffer)
        
        return avg_pitch, avg_roll

    def check_head_tilt(self, pitch_thresh=45, roll_thresh=45):
        """Returns True if head tilted beyond threshold"""
        pitch, roll = self.get_orientation()
        is_tilted = abs(pitch) > pitch_thresh or abs(roll) > roll_thresh
        
        # LED control
        if is_tilted:
            self.led.on()
        else:
            self.led.off()
            
        return is_tilted

    def continuous_monitoring(self):
        try:
            while True:
                pitch, roll = self.get_orientation()
                tilted = self.check_head_tilt()
                
                print(f"Pitch: {pitch:.1f}° | Roll: {roll:.1f}° | Tilted: {'YES' if tilted else 'NO'}")
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.led.off()
            print("Head tracking stopped")