import serial
import pynmea2
import time

def get_gps_location(port='/dev/serial0', baudrate=9600, timeout=30):
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            start_time = time.time()
            while time.time() - start_time < timeout:
                data = ser.readline().decode('ascii', errors='ignore').strip()
                print(data)
                if data.startswith('$GPGGA'):
                    msg = pynmea2.parse(data)
                    if msg.latitude and msg.longitude:
                        return {
                            'latitude': msg.latitude,
                            'longitude': msg.longitude,
                            'altitude': msg.altitude,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
            return None
    except Exception as e:
        print(f"GPS Error: {e}")
        return None

if __name__ == '__main__':
    print("Waiting for GPS fix... (Go outdoors!)")
    location = get_gps_location()
    if location:
        print(f"GPS Location Acquired:\n{location}")
    else:
        print("Failed to get GPS fix after 30 seconds")