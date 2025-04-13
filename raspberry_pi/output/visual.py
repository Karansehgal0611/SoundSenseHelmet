import time
import RPi.GPIO as GPIO


class LEDController:
    def __init__(self, pin=17):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        self.off()  # Start with LED off

    def on(self):
        GPIO.output(self.pin, GPIO.HIGH)
    
    def off(self):
        GPIO.output(self.pin, GPIO.LOW)
    
    def blink(self, times=3, speed=0.5):
        for _ in range(times):
            self.on()
            time.sleep(speed)
            self.off()
            time.sleep(speed)
    
    def emergency_pattern(self):
        """SOS pattern: ... --- ..."""
        self.blink(3, 0.3)  # Fast
        time.sleep(0.5)
        self.blink(3, 0.7)  # Slow
        time.sleep(0.5)
        self.blink(3, 0.3)  # Fast

    def police_siren_pattern(self):
        """Alternating fast/slow flashes"""
        for _ in range(4):
            self.blink(2, 0.1)
            time.sleep(0.1)
            self.blink(1, 0.5)
    
    def cleanup(self):
        GPIO.cleanup()
    
    @staticmethod
    def create_mpu_led():
        """Factory method for MPU-specific LED"""
        return LEDController(config.MPU_LED_PIN)



