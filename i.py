import smbus2
bus = smbus2.SMBus(1)
try:
    # Try both addresses
    for addr in [0x68, 0x69]:
        whoami = bus.read_byte_data(addr, 0x75)  # WHO_AM_I register
        print(f"0x{addr:02x} returned: 0x{whoami:02x}")  # Should be 0x68 or 0x71
except Exception as e:
    print(f"I2C failed: {e}")