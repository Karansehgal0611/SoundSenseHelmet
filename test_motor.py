import RPi.GPIO as GPIO
import time

MOTOR_PIN = 12  # GPIO 18 (Pin 12)
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(MOTOR_PIN, GPIO.HIGH)
        print("ON")  # Vibrate ON
        time.sleep(0.5)
        GPIO.output(MOTOR_PIN, GPIO.LOW)
        print("OFF")   # Vibrate OFF
        time.sleep(2)
except KeyboardInterrupt:
    GPIO.cleanup()