# backtest_rl.py (Updated for consistency with train.py and batteryEnv.py)
# Backtests a specific MLflow PPO model version on historical/test data, with Oracle comparison.
# Usage: python backtest_rl.py --model_version 1 --data_path /path/to/data.csv

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt
import os
from batteryEnv import BatteryEnv  # Import full env for spaces/logic

# Constants (match train.py and batteryEnv.py)
STORAGE_CAPACITY_KWH = 215.0
CHARGE_RATE_KW = 40.0
EFFICIENCY = 0.96
PRICE_COLUMN = 'Price (EUR)'
HOUR_COLUMN = 'Hour'
DAY_OF_WEEK_COLUMN = 'DayOfWeek'
MONTH_COLUMN = 'Month'
PRICE_AVG24_COLUMN = 'price_rolling_avg_24h'
DATE_COLUMN = 'Date'

def load_data(data_path, lstm_wrapper):
    df = pd.read_csv(data_path, sep=';', decimal=',')
    df.rename(columns={'Price (EUR)': PRICE_COLUMN, 'Date': DATE_COLUMN}, inplace=True)
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors='coerce')
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format='%m/%d/%y', errors='coerce')
    df.dropna(inplace=True)
    df.set_index(DATE_COLUMN, inplace=True)
    df[PRICE_AVG24_COLUMN] = df[PRICE_COLUMN].rolling(window=24, min_periods=1).mean()
    df['DayOfWeek'] = df.index.weekday
    df['Month'] = df.index.month

    # Precompute 24h forecasts (match train.py)
    forecasts = []
    seq_len = 24
    for i in range(len(df) - seq_len):
        seq = df[PRICE_COLUMN].values[i:i+seq_len].reshape(1, seq_len)
        forecast_seq = lstm_wrapper.predict(seq)[0]
        forecasts.append(forecast_seq.tolist())
    forecasts = [[0.0] * 24] * seq_len + forecasts
    df['forecasted_prices'] = forecasts
    return df

def calculate_oracle_profit(df, max_battery_capacity, charge_discharge_rate):
    # Improved Oracle: Simulate perfect foresight with capacity constraints (greedy buy low/sell high, respecting rates)
    prices = df[PRICE_COLUMN].values
    battery_charge = max_battery_capacity / 2
    total_profit = 0.0
    buy_threshold = np.percentile(prices, 30)  # Buy below 30th percentile (tunable)
    sell_threshold = np.percentile(prices, 70)  # Sell above 70th percentile
    for price in prices:
        price_kwh = price / 1000.0
        if price < buy_threshold and battery_charge < max_battery_capacity:
            amount = min(charge_discharge_rate, max_battery_capacity - battery_charge)
            total_profit -= price_kwh * amount
            battery_charge += amount * EFFICIENCY
        elif price > sell_threshold and battery_charge > 0:
            amount = min(charge_discharge_rate, battery_charge)
            total_profit += price_kwh * amount
            battery_charge -= amount
    return total_profit

def main(model_version, data_path):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    lstm_uri = "models:/BESS_Price_Forecaster/Latest"
    lstm_wrapper = mlflow.pyfunc.load_model(lstm_uri)
    df = load_data(data_path, lstm_wrapper)

    model_uri = f"models:/BatteryPPOModel/{model_version}"
    model_wrapper = mlflow.pyfunc.load_model(model_uri)

    # Simulation (match evaluate_all_models.py)
    battery_charge = STORAGE_CAPACITY_KWH / 2
    total_profit = 0.0
    log = []
    for idx, row in df.iterrows():
        battery_percent = battery_charge / STORAGE_CAPACITY_KWH
        hour_sin = np.sin(2 * np.pi * row[HOUR_COLUMN] / 24)
        hour_cos = np.cos(2 * np.pi * row[HOUR_COLUMN] / 24)
        day_sin = np.sin(2 * np.pi * row[DAY_OF_WEEK_COLUMN] / 6)
        day_cos = np.cos(2 * np.pi * row[DAY_OF_WEEK_COLUMN] / 6)
        month_sin = np.sin(2 * np.pi * row[MONTH_COLUMN] / 12)
        month_cos = np.cos(2 * np.pi * row[MONTH_COLUMN] / 12)
        obs_raw = np.array([
            row[PRICE_COLUMN], hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos,
            row[PRICE_AVG24_COLUMN], battery_percent
        ] + row['forecasted_prices'], dtype=np.float32)[None, :]
        action = model_wrapper.predict(obs_raw)[0]
        action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action, 'HOLD')

        price_kwh = row[PRICE_COLUMN] / 1000.0
        profit_change = 0.0
        if action_str == 'BUY' and battery_charge < STORAGE_CAPACITY_KWH:
            amount = min(CHARGE_RATE_KW, STORAGE_CAPACITY_KWH - battery_charge)
            profit_change = -price_kwh * amount
            battery_charge += amount * EFFICIENCY
        elif action_str == 'SELL' and battery_charge > 0:
            amount = min(CHARGE_RATE_KW, battery_charge)
            profit_change = price_kwh * amount
            battery_charge -= amount
        total_profit += profit_change
        log.append({'timestamp': idx, 'price': row[PRICE_COLUMN], 'action': action_str, 'profit': profit_change, 'cum_profit': total_profit, 'battery_kwh': battery_charge})

    oracle_profit = calculate_oracle_profit(df, STORAGE_CAPACITY_KWH, CHARGE_RATE_KW)
    performance_score = (total_profit / oracle_profit * 100) if oracle_profit > 0 else 0
    print(f"RL Profit: €{total_profit:.2f}, Oracle: €{oracle_profit:.2f}, Score: {performance_score:.2f}%")

    # Visualize
    log_df = pd.DataFrame(log)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(log_df['timestamp'], log_df['price'], label='Price', color='gray')
    ax1.scatter(log_df[log_df['action']=='BUY']['timestamp'], log_df[log_df['action']=='BUY']['price'], color='green', label='Buy')
    ax1.scatter(log_df[log_df['action']=='SELL']['timestamp'], log_df[log_df['action']=='SELL']['price'], color='red', label='Sell')
    ax2 = ax1.twinx()
    ax2.plot(log_df['timestamp'], log_df['cum_profit'], label='Cum Profit', color='blue')
    ax2.plot(log_df['timestamp'], log_df['battery_kwh'], label='Battery kWh', color='orange', linestyle='--')
    plt.title('Backtest: Actions, Profit, Battery')
    plt.savefig('backtest_plot.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", default="Latest", help="MLflow model version")
    parser.add_argument("--data_path", default="/Users/chavdarbilyanski/energyscale/src/ml/data/combine/2025_1H_output_with_features.csv")
    args = parser.parse_args()
    main(args.model_version, args.data_path)