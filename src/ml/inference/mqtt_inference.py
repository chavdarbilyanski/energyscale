# src/ml/inference/mqtt_inference.py
# Standalone script for real-time MQTT-based inference using RL model for BESS control.
# Subscribes to edge device data, constructs obs (using SoC from msg + maintained price/forecast data),
# infers action, publishes command. Maintains price history via periodic API fetches.
# Run locally: python mqtt_inference.py
# For GCP: Deploy on Compute Engine VM (persistent), set env vars for MLflow URI, MQTT broker, API keys.
# Requirements: Add 'paho-mqtt requests' to requirements.txt; pip install.

import os
import json
import mlflow
import numpy as np
from datetime import datetime
import time
import threading
import paho.mqtt.client as mqtt
from collections import deque
import requests  # For price API

# --- Config (Override via env vars for GCP/prod) ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))
SUBSCRIBE_TOPIC = os.environ.get("SUBSCRIBE_TOPIC", "edge/#")  # Wildcard for all edge devices
COMMAND_TOPIC_PREFIX = "cmd"  # New: Distinct prefix for commands (server-to-edge)
COMMAND_TOPIC_SUFFIX = "/channel/command"  # Suffix for dynamic command topic (e.g., cmd/edge0/channel/command)
PRICE_API_URL = os.environ.get("PRICE_API_URL", "https://electricity-price.p.rapidapi.com/v1/spot")  # Example; set to actual endpoint
PRICE_API_HEADERS = {
    "X-RapidAPI-Key": os.environ.get("RAPIDAPI_KEY", "your-key-here"),  # Set in .env or GCP secrets
    "X-RapidAPI-Host": "electricity-price.p.rapidapi.com"
}
PRICE_API_PARAMS = {"country": "DE"}  # e.g., Germany; adjust for region
PRICE_FETCH_INTERVAL = 3600  # Seconds (hourly)
CHARGE_RATE_W = 40000  # 40 kW in W (match JSON units)
STORAGE_CAPACITY_KWH = 215.0  # From train.py (for potential checks; not used here)

# Load models from MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_wrapper = mlflow.pyfunc.load_model("models:/BatteryPPOModel/Latest")
lstm_wrapper = mlflow.pyfunc.load_model("models:/BESS_Price_Forecaster/Latest")

# Global state for price data (thread-safe access if needed; simple for now)
price_history = deque(maxlen=24)  # Last 24 prices (EUR/MWh)
current_price = 0.0
rolling_avg = 0.0
forecasted_prices = [0.0] * 24

def fetch_price_loop_from_api():
    """Thread to periodically fetch/update current price, history, rolling avg, forecasts."""
    global current_price, rolling_avg, forecasted_prices
    while True:
        try:
            response = requests.get(PRICE_API_URL, headers=PRICE_API_HEADERS, params=PRICE_API_PARAMS)
            response.raise_for_status()
            data = response.json()
            current_price = data.get('price', 0.0)  # Assume {'price': float} response; adjust per API
            price_history.append(current_price)
            if len(price_history) > 0:
                rolling_avg = np.mean(price_history)
            if len(price_history) == 24:
                forecasted_prices = lstm_wrapper.predict(np.array(price_history)[None, :])[0].tolist()
            print(f"Fetched price: {current_price} EUR/MWh, Rolling Avg: {rolling_avg}, Forecast: {forecasted_prices[:5]}...")
        except Exception as e:
            print(f"Price fetch error: {e}")
        time.sleep(PRICE_FETCH_INTERVAL)

def fetch_price_loop():
    """Thread to periodically fetch/update current price, history, rolling avg, forecasts.
    Mocked version: Uses static/simulated prices instead of real API for local MQTT integration.
    """
    global current_price, rolling_avg, forecasted_prices
    # Static mock prices (EUR/MWh): Cycle through a list for simulation (e.g., daily pattern)
    mock_prices = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
                   150.0, 160.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0,
                   110.0, 100.0, 90.0, 80.0, 70.0, 60.0]  # 24h cycle; repeat
    price_idx = 0
    
    # Prefill history with initial static data to enable forecasts quickly
    initial_history = mock_prices[:24]  # Full 24 for immediate LSTM if needed
    for p in initial_history:
        price_history.append(p)
    rolling_avg = np.mean(price_history) if len(price_history) > 0 else 0.0
    try:
        forecasted_prices = lstm_wrapper.predict(np.array(initial_history)[None, :])[0].tolist()
    except Exception as e:
        print(f"Initial forecast mock error (using zeros): {e}")
        forecasted_prices = [0.0] * 24  # Fallback if LSTM not ready
    
    print(f"Initial mock setup: Rolling Avg: {rolling_avg}, Forecast: {forecasted_prices[:5]}...")
    
    while True:
        try:
            # Mock API: Get next price from cycle (no internet)
            current_price = mock_prices[price_idx % len(mock_prices)]
            price_idx += 1
            
            price_history.append(current_price)
            rolling_avg = np.mean(price_history) if len(price_history) > 0 else 0.0
            
            if len(price_history) == 24:  # Update forecast only when full window
                try:
                    forecasted_prices = lstm_wrapper.predict(np.array(list(price_history))[None, :])[0].tolist()
                except Exception as e:
                    print(f"Forecast mock error (keeping previous): {e}")
                    # Keep previous or fallback to avg-based mock forecast
                    forecasted_prices = [rolling_avg] * 24
            
            print(f"Mock fetched price: {current_price} EUR/MWh, Rolling Avg: {rolling_avg}, Forecast: {forecasted_prices[:5]}...")
        except Exception as e:
            print(f"Price mock error: {e}")
        time.sleep(PRICE_FETCH_INTERVAL)

def compute_cyclical_features(hour, dayofweek, month):
    """Compute sin/cos for time features (match batteryEnv.py)."""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * dayofweek / 6)
    day_cos = np.cos(2 * np.pi * dayofweek / 6)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to MQTT broker.")
        client.subscribe(SUBSCRIBE_TOPIC)
    else:
        print(f"Connection failed: {reason_code}")

def on_message(client, userdata, msg):
    """Handle incoming edge device JSON, infer action, publish command to dynamic topic."""
    try:
        # Extract device name from topic (e.g., 'edge/edge0/channel/data' -> device_name = 'edge0')
        topic_parts = msg.topic.split('/')
        if len(topic_parts) >= 2:
            device_name = topic_parts[1]
        else:
            raise ValueError("Invalid topic format; cannot extract device name.")

        data = json.loads(msg.payload.decode())
        last_update = data['lastUpdate']
        soc = data['channels']['_sum']['EssSoc']
        battery_percent = soc / 100.0

        # Parse timestamp for time features
        dt = datetime.fromisoformat(last_update.rstrip('Z'))
        hour = dt.hour
        dayofweek = dt.weekday()
        month = dt.month
        hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos = compute_cyclical_features(hour, dayofweek, month)

        # Construct observation (use latest price/forecast data + msg SoC/time)
        obs = np.array([
            current_price,
            hour_sin, hour_cos,
            day_sin, day_cos,
            month_sin, month_cos,
            rolling_avg,
            battery_percent
        ] + forecasted_prices, dtype=np.float32)

        # Inference
        action = model_wrapper.predict(obs[None, :])[0]
        action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action, 'HOLD')

        # Map to BESS command (e.g., SetActivePower: negative for charge/BUY, positive for discharge/SELL)
        if action_str == 'BUY':
            power = -CHARGE_RATE_W  # Charge (negative)
        elif action_str == 'SELL':
            power = CHARGE_RATE_W  # Discharge (positive)
        else:
            power = 0

        command = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "channels": {
                "ess0": {
                    "SetActivePower": power
                }
            }
        }

        # Construct dynamic publish topic (e.g., cmd/edge0/channel/command)
        publish_topic = f"{COMMAND_TOPIC_PREFIX}/{device_name}{COMMAND_TOPIC_SUFFIX}"
        client.publish(publish_topic, json.dumps(command), qos=1)
        print(f"Published command: {command} to {publish_topic}")
    except Exception as e:
        print(f"Message processing error: {e}")

# Start MQTT client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# Start price fetch thread
price_thread = threading.Thread(target=fetch_price_loop, daemon=True)
price_thread.start()

# Keep script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    client.loop_stop()