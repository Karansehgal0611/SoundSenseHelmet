import RPi.GPIO as GPIO
import time
import Adafruit_MCP3008

# MCP3008 setup
mcp = Adafruit_MCP3008.MCP3008(
    CLK=11, CS=8, MISO=9, MOSI=10)

def read_fsr():
    raw_value = mcp.read_adc(0)  # Read CH0
    voltage = (raw_value / 1023.0) * 3.3  # Convert to voltage
    return raw_value, voltage

try:
    while True:
        raw, voltage = read_fsr()
        print(f"Raw: {raw}, Voltage: {voltage:.2f}V")
        time.sleep(0.5)
except KeyboardInterrupt:
    GPIO.cleanup()

    