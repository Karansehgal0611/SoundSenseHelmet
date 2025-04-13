import time
from raspberry_pi.output.visual import LEDController

led = LEDController()

try:
    print("Testing LED - 3 slow blinks")
    led.blink(times=3, speed=1)
    
    print("Testing rapid alert pattern")
    for _ in range(5):
        led.on()
        time.sleep(0.1)
        led.off()
        time.sleep(0.1)
        
finally:
    led.cleanup()