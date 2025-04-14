# raspberry_pi/output/haptic.py
import RPi.GPIO as GPIO
import time
from threading import Thread

class HapticController:
    def __init__(self, motor_pin=12):
        """
        Initialize vibration motor controller
        :param motor_pin: GPIO pin connected to transistor base
        """
        self.motor_pin = motor_pin
        self._running = False
        self._thread = None
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.motor_pin, GPIO.OUT)
        GPIO.output(self.motor_pin, GPIO.LOW)
        
    def _vibrate_pattern(self, duration=0.5, times=3, interval=0.3):
        """Internal method to run vibration pattern"""
        for _ in range(times):
            GPIO.output(self.motor_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.motor_pin, GPIO.LOW)
            if _ < times - 1:  # No interval after last vibration
                time.sleep(interval)
    
    def vibrate(self, duration=0.5, times=3, blocking=False):
        """
        Trigger vibration pattern
        :param duration: Seconds per vibration pulse
        :param times: Number of pulses
        :param blocking: If True, waits for pattern to complete
        """
        if blocking:
            self._vibrate_pattern(duration, times)
        else:
            if self._thread and self._thread.is_alive():
                self._thread.join(0.1)
            self._thread = Thread(target=self._vibrate_pattern, 
                                 args=(duration, times))
            self._thread.start()
    
    def emergency_sos(self):
        """Play SOS pattern (... --- ...)"""
        pattern = [
            (0.2, 3),  # Short pulses for '...'
            (0.5, 3),   # Long pulses for '---'
            (0.2, 3)    # Short pulses for '...'
        ]
        for duration, times in pattern:
            self.vibrate(duration, times, blocking=True)
            time.sleep(0.3)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self._thread and self._thread.is_alive():
            self._thread.join()
        GPIO.output(self.motor_pin, GPIO.LOW)
        GPIO.cleanup()

# Example usage
if __name__ == "__main__":
    try:
        motor = HapticController()
        print("Testing short vibration...")
        motor.vibrate(duration=0.3, times=2)
        time.sleep(2)
        print("Testing SOS pattern...")
        motor.emergency_sos()
    finally:
        motor.cleanup()