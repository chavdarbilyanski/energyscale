# src/web/webbidder/services.py

import os
import logging
from io import StringIO
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# Global caching for the loaded pyfunc model (avoids reloading on every request)
_model_wrapper = None

def load_rl_model_and_env():
    global _model_wrapper
    if _model_wrapper is not None:
        return _model_wrapper
    
    logger.info("Loading RL model from MLflow by tag...")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient()
    # Search for version with tag 'stage: Production'
    filter_string = "name='BatteryPPOModel' AND tag.stage = 'Production'"
    versions = client.search_model_versions(filter_string)
    if not versions:
        raise ValueError("No Production-tagged model found.")
    
    # Pick the latest (or sort by creation_time if multiple)
    latest_prod_version = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)[0]
    model_uri = f"models:/BatteryPPOModel/{latest_prod_version.version}"
    
    _model_wrapper = mlflow.pyfunc.load_model(model_uri)
    return _model_wrapper

# --- Main Service Function for RL Model Simulation ---
def run_rl_model_simulation(csv_file, max_battery_capacity, charge_discharge_rate):
    """Run RL model simulation for BESS profit maximization."""
    model_wrapper = load_rl_model_and_env()  # Load the pyfunc wrapper (handles PPO and VecNormalize)
    
    battery_charge = 0.0
    total_profit = 0.0
    all_results = []
    
    csv_data = csv_file.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data), sep=';', decimal=',')
    
    # Preprocessing logic
    df.rename(columns={'Price (EUR)': 'price', 'Date': 'timestamp', 'Hour': 'hour'}, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y', errors='coerce')
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("CSV is empty after cleaning.")

    df['DayOfWeek'] = df['timestamp'].dt.weekday
    df['Month'] = df['timestamp'].dt.month
    df['price_rolling_avg_24h'] = df['price'].rolling(window=24, min_periods=1).mean()
    
    # Inference Loop
    for _, row in df.iterrows():
        battery_percent = (battery_charge / max_battery_capacity) if max_battery_capacity > 0 else 0.0
        price_mwh = row['price']
        price_kwh = price_mwh / 1000.0

        # Compute cyclical features matching BatteryEnv
        hour_sin = np.sin(2 * np.pi * row['hour'] / 24)
        hour_cos = np.cos(2 * np.pi * row['hour'] / 24)
        day_sin = np.sin(2 * np.pi * row['DayOfWeek'] / 6)
        day_cos = np.cos(2 * np.pi * row['DayOfWeek'] / 6)
        month_sin = np.sin(2 * np.pi * row['Month'] / 12)
        month_cos = np.cos(2 * np.pi * row['Month'] / 12)

        obs_raw = np.array([
            price_mwh,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
            month_sin,
            month_cos,
            row['price_rolling_avg_24h'],
            battery_percent
        ], dtype=np.float32)[None, :]  # Shape: (1, 9) for pyfunc
        
        action = model_wrapper.predict(obs_raw)[0]  # Pyfunc handles normalization and prediction
        action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action, 'HOLD')
        
        # Capture state *before* action for logging
        battery_charge_before = battery_charge
        battery_percent_before = battery_percent
        
        profit_change = 0.0
        if action_str == 'BUY' and battery_charge < max_battery_capacity:
            amount = min(charge_discharge_rate, max_battery_capacity - battery_charge)
            profit_change = -price_kwh * amount
            battery_charge += amount
        elif action_str == 'SELL' and battery_charge > 0:
            amount = min(charge_discharge_rate, battery_charge)
            profit_change = price_kwh * amount
            battery_charge -= amount
        
        total_profit += profit_change
        
        all_results.append({
            'timestamp': row['timestamp'].strftime('%d/%m/%Y'),
            'hour': int(row['hour']),
            'price': price_kwh,
            'action': action_str,
            'profit_change': profit_change,
            'total_profit': total_profit,
            'battery_charge': battery_charge_before,
            'battery_percent': battery_percent_before * 100.0,
        })
        
    return total_profit, all_results

# --- Main Service Function for the Correct Oracle Model ---
def calculate_globally_optimal_profit(csv_file, max_battery_capacity, charge_discharge_rate):
    """
    Calculates the true maximum possible profit by pairing the cheapest
    buy opportunities with the most expensive sell opportunities.
    """
    logger.info("Calculating globally optimal profit (True Oracle)...")
    
    csv_file.seek(0)  # Reset file pointer
    csv_data = csv_file.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data), sep=';', decimal=',')
    
    df.rename(columns={'Price (EUR)': 'price', 'Date': 'timestamp', 'Hour': 'hour'}, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(inplace=True)
    if df.empty:
        return 0.0

    # Create lists of buy/sell slots
    buy_prices = []
    sell_prices = []
    slots_per_hour = int(charge_discharge_rate)
    
    for price in df['price']:
        buy_prices.extend([price] * slots_per_hour)
        sell_prices.extend([price] * slots_per_hour)
        
    # Sort opportunities
    buy_prices.sort()
    sell_prices.sort(reverse=True)
    
    # Calculate profit by pairing
    total_profit_mwh = 0.0
    for buy_price, sell_price in zip(buy_prices, sell_prices):
        if sell_price > buy_price:
            total_profit_mwh += (sell_price - buy_price)
        else:
            break
            
    # Convert to EUR/kWh
    total_profit_kwh = total_profit_mwh / 1000.0
    
    logger.info(f"[True Oracle] Max possible profit: {total_profit_kwh:.2f}")
    return total_profit_kwh