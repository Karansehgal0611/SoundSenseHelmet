# raspberry_pi/utils/alerts.py
import requests
import json
import logging
from threading import Thread

logger = logging.getLogger(__name__)
FLASK_SERVER_URL = "http://[CLIENT_IP]:5001/api/accident"

def send_alert(location_data):
    """Send accident alert to Flask server"""
    def _send_async():
        try:
            payload = {
                'latitude': location_data['latitude'],
                'longitude': location_data['longitude'],
                'timestamp': location_data['timestamp']
            }
            response = requests.post(
                FLASK_SERVER_URL + "/alert",
                json=payload,
                timeout=5
            )
            if response.status_code != 200:
                logger.error(f"Alert failed: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    Thread(target=_send_async).start()