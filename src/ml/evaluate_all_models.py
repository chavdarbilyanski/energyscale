# evaluate_mlflow_models.py
# This script evaluates all versions of the BatteryPPOModel from MLflow against a test CSV dataset for BESS profit maximization.
# It computes RL-simulated profit for each model version and compares it to the true optimal (Oracle) profit.
# Designed for quick, static runs locally or on GCP (e.g., via Cloud Run or VM cron jobs).
# Usage: python evaluate_mlflow_models.py --test_csv_path /path/to/Test.csv --tracking_uri http://127.0.0.1:5000
#        (Defaults: Test.csv in cwd, local MLflow URI)
# For BESS + Solar Park extensions: Add solar_kwh to CSV/obs for hybrid profit calcs.

import argparse
import logging
import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from io import StringIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PRICE_COLUMN = 'price'  # Matches renaming in load_test_data; use 'Price (EUR)' if before rename
STORAGE_CAPACITY_KWH = 215.0  # Already defined, but ensure consistency
CHARGE_RATE_KW = 40.0
EFFICIENCY = 0.96  # Add this to match batteryEnv.py for realistic losses
DEGRADATION_COST = 0.01

# Hardcoded BESS params (match train.py; override via args if needed for GCP flexibility)
STORAGE_CAPACITY_KWH = 215.0
CHARGE_RATE_KW = 40.0

def load_test_data(test_csv_path, lstm_wrapper):
    """Load and preprocess Test.csv for BESS simulation."""
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")
    
    with open(test_csv_path, 'r') as f:
        csv_data = f.read()
    
    df = pd.read_csv(StringIO(csv_data), sep=';', decimal=',')
    df.rename(columns={'Price (EUR)': 'price', 'Date': 'timestamp', 'Hour': 'hour'}, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%y', errors='coerce')  # Updated to handle four-digit year (e.g., 1/1/2025)
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Test CSV is empty after cleaning.")

    df['DayOfWeek'] = df['timestamp'].dt.weekday
    df['Month'] = df['timestamp'].dt.month
    df['price_rolling_avg_24h'] = df['price'].rolling(window=24, min_periods=1).mean()

    # Compute full 24h forecasted_prices (sliding window as list)
    forecasts = []
    seq_len = 24
    for i in range(len(df) - seq_len):
        seq = df['price'].values[i:i+seq_len].reshape(1, seq_len)
        forecast_seq = lstm_wrapper.predict(seq)[0]  # Full (24,) array
        forecasts.append(forecast_seq.tolist())  # Convert to list
    forecasts = [[0.0] * 24] * seq_len + forecasts  # Pad beginning with zero lists
    df['forecasted_prices'] = forecasts
    
    return df

def run_model_simulation(model_wrapper, df, max_battery_capacity, charge_discharge_rate):
    """Simulate BESS profit for a loaded MLflow pyfunc model (matches services.py logic)."""
    battery_charge = 0.0
    total_profit = 0.0
    
    for _, row in df.iterrows():
        battery_percent = (battery_charge / max_battery_capacity) if max_battery_capacity > 0 else 0.0
        price_mwh = row['price']
        price_kwh = price_mwh / 1000.0

        # Compute cyclical features (match BatteryEnv for consistent obs)
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
        ] + row['forecasted_prices'], dtype=np.float32)[None, :]  # Now 33 dims
        
        action = model_wrapper.predict(obs_raw)[0]
        action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action, 'HOLD')
        
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
    
    return total_profit

def calculate_oracle_profit(df, max_battery_capacity, charge_discharge_rate):
    """Constrained optimal profit (Oracle) via DP: Respects sequence, capacity, efficiency, degradation per transaction."""
    prices = df['price'].values
    T = len(prices)
    capacity = int(max_battery_capacity)
    rate = int(charge_discharge_rate)
    
    # DP table: dp[t][b] = max profit up to timestep t with battery level b
    dp = np.full((T + 1, capacity + 1), -np.inf)
    start_charge = capacity // 2  # Match env initial state
    dp[0][start_charge] = 0.0
    
    for t in range(T):
        price_kwh = prices[t] / 1000.0
        for b in range(capacity + 1):
            if dp[t][b] == -np.inf:
                continue
            # HOLD: No change, no degradation
            dp[t + 1][b] = max(dp[t + 1][b], dp[t][b])
            # BUY: If space, apply efficiency on charge, degradation
            if b < capacity:
                amount = min(rate, capacity - b)
                cost = amount * price_kwh
                new_b = min(int(b + amount * EFFICIENCY), capacity)  # Efficiency loss
                new_profit = dp[t][b] - cost - DEGRADATION_COST
                dp[t + 1][new_b] = max(dp[t + 1][new_b], new_profit)
            # SELL: If charged, degradation
            if b > 0:
                amount = min(rate, b)
                revenue = amount * price_kwh
                new_b = b - amount
                new_profit = dp[t][b] + revenue - DEGRADATION_COST
                dp[t + 1][new_b] = max(dp[t + 1][new_b], new_profit)
    
    max_profit = np.max(dp[T])
    if max_profit == -np.inf:
        raise ValueError("No feasible Oracle profit calculated.")
    return max_profit

def calculate_oracle_profit_uncostrained_by_time_capacity(df, max_battery_capacity, charge_discharge_rate):
    """Compute true optimal profit (Oracle) by pairing cheapest buys with expensive sells, with efficiency and degradation per transaction."""
    buy_prices = []
    sell_prices = []
    slots_per_hour = int(charge_discharge_rate)
    
    for price in df['price']:
        buy_prices.extend([price] * slots_per_hour)
        sell_prices.extend([price] * slots_per_hour)
    
    buy_prices.sort()
    sell_prices.sort(reverse=True)
    
    total_profit = 0.0
    for buy_price, sell_price in zip(buy_prices, sell_prices):
        # Profit per 1kWh cycle: (sell * efficiency - buy) / 1000 - 2 * degradation (for buy + sell)
        profit_per_cycle = ((sell_price * EFFICIENCY - buy_price) / 1000.0) - 2 * DEGRADATION_COST
        if profit_per_cycle > 0:
            total_profit += profit_per_cycle
        else:
            break
    
    return total_profit

def calculate_oracle_profit2(df, max_battery_capacity, charge_discharge_rate):
    """Exact optimal profit (Oracle) via DP with perfect foresight, respecting constraints."""
    prices = df['price'].values
    T = len(prices)
    capacity = int(max_battery_capacity)  # e.g., 215
    rate = int(charge_discharge_rate)  # e.g., 40
    efficiency = EFFICIENCY
    degradation = DEGRADATION_COST  # Or 0 for looser bound

    # DP table: dp[t][b] = max profit after t steps with battery level b (0 to capacity)
    dp = np.full((T + 1, capacity + 1), -np.inf)
    start_charge = capacity // 2
    dp[0][start_charge] = 0.0

    for t in range(T):
        price_kwh = prices[t] / 1000.0
        for b in range(capacity + 1):
            if dp[t][b] == -np.inf:
                continue
            # HOLD: No change
            dp[t + 1][b] = max(dp[t + 1][b], dp[t][b])
            # BUY: If space available
            if b < capacity:
                amount = min(rate, capacity - b)
                cost = amount * price_kwh
                new_b = min(int(b + amount * efficiency), capacity)  # Apply efficiency, cap at max
                new_profit = dp[t][b] - cost - degradation
                dp[t + 1][new_b] = max(dp[t + 1][new_b], new_profit)
            # SELL: If charge available
            if b > 0:
                amount = min(rate, b)
                revenue = amount * price_kwh
                new_b = b - amount
                new_profit = dp[t][b] + revenue - degradation
                dp[t + 1][new_b] = max(dp[t + 1][new_b], new_profit)

    # Return the max profit across all final battery states
    max_profit = np.max(dp[T])
    if max_profit == -np.inf:
        raise ValueError("No feasible Oracle profit calculated.")
    return max_profit

def main(tracking_uri, test_csv_path, model_name="BatteryPPOModel"):
    """Main evaluation loop: Fetch models, simulate, compare to Oracle."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    # Get all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}' in MLflow.")
    
    lstm_uri = "models:/BESS_Price_Forecaster/Latest"
    lstm_wrapper = mlflow.pyfunc.load_model(lstm_uri)
 
    df = load_test_data(test_csv_path, lstm_wrapper)
    oracle_profit = calculate_oracle_profit(df, STORAGE_CAPACITY_KWH, CHARGE_RATE_KW)
    logger.info(f"Oracle Profit: €{oracle_profit:.2f}")
    
    results = []
    for version in versions:
        model_uri = f"models:/{model_name}/{version.version}"
        logger.info(f"Evaluating model version {version.version} (Run ID: {version.run_id}) from {model_uri}")
        
        try:
            model_wrapper = mlflow.pyfunc.load_model(model_uri)
            rl_profit = run_model_simulation(model_wrapper, df, STORAGE_CAPACITY_KWH, CHARGE_RATE_KW)
            performance_score = (rl_profit / oracle_profit) * 100 if oracle_profit > 0 else 0
            results.append({
                'version': version.version,
                'run_id': version.run_id,
                'rl_profit': rl_profit,
                'oracle_profit': oracle_profit,
                'performance_score': performance_score
            })
            logger.info(f"Version {version.version}: RL Profit €{rl_profit:.2f}, Score {performance_score:.2f}%")
        except Exception as e:
            logger.error(f"Error evaluating version {version.version}: {e}")
    
    # Print summary table
    print("\n--- Evaluation Summary ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLflow BESS models on Test.csv for profit maximization.")
    parser.add_argument("--tracking_uri", default="http://127.0.0.1:5000", help="MLflow tracking URI (local or gs:// for GCP).")
    parser.add_argument("--test_csv_path", default="/Users/chavdarbilyanski/energyscale/src/ml/data/combine/2025_1H_output_with_features.csv", help="Path to Test.csv dataset (defaults to Test.csv in cwd).")
    args = parser.parse_args()
    
    main(args.tracking_uri, args.test_csv_path)