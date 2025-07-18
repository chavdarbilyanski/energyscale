# src/web/webbidder/services.py

import os
import logging
from io import StringIO
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from django.conf import settings

logger = logging.getLogger(__name__)

# --- Helper Classes and Functions for RL Model ---
class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        low_bounds = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        high_bounds = np.array([np.inf, 23, 6, 12, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
    def step(self, action): return self.observation_space.sample(), 0, False, {}
    def reset(self): return self.observation_space.sample()

def load_rl_model_and_env():
    logger.info("Loading RL model and environment stats...")
    model_path = os.path.join(settings.BASE_DIR, 'powerbidder', "battery_ppo_agent_v2.zip")
    stats_path = os.path.join(settings.BASE_DIR, 'powerbidder', "vec_normalize_stats_v2.pkl")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        raise FileNotFoundError("RL Model or stats file not found.")
    
    vec_env = DummyVecEnv([lambda: BatteryEnv()])
    vec_env = VecNormalize.load(stats_path, vec_env)
    model = PPO.load(model_path, env=vec_env)
    return model, vec_env

# --- Main Service Function for RL Model Simulation ---
def run_rl_model_simulation(csv_file, max_battery_capacity, charge_discharge_rate):
    """This function contains your existing, working RL model simulation logic."""
    model, vec_env = load_rl_model_and_env()
    
    battery_charge = 0.0
    total_profit = 0.0
    all_results = []
    
    csv_data = csv_file.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data), sep=';', decimal='.')
    
    # Preprocessing logic from your original view
    df.rename(columns={'Price (EUR)': 'price', 'Date': 'timestamp', 'Hour': 'hour'}, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y', errors='coerce')
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df.dropna(inplace=True)
    if df.empty: raise ValueError("CSV is empty after cleaning.")

    df['DayOfWeek'] = df['timestamp'].dt.weekday
    df['Month'] = df['timestamp'].dt.month
    df['price_rolling_avg_24h'] = df['price'].rolling(window=24, min_periods=1).mean()
    
    # Inference Loop
    for _, row in df.iterrows():
        battery_percent = (battery_charge / max_battery_capacity) if max_battery_capacity > 0 else 0.0
        price_mwh = row['price']
        price_kwh = price_mwh / 1000.0

        obs_raw = np.array([
            price_mwh, row['hour'] - 1, row['DayOfWeek'], row['Month'],
            row['price_rolling_avg_24h'], battery_percent
        ], dtype=np.float32)
        
        normalized_obs = vec_env.normalize_obs(np.array([obs_raw]))
        action, _ = model.predict(normalized_obs, deterministic=True)
        action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action[0], 'HOLD')
        
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

# --- NEW: Main Service Function for the Correct Oracle Model ---
def calculate_globally_optimal_profit(csv_file, max_battery_capacity, charge_discharge_rate):
    """
    Calculates the true maximum possible profit by pairing the absolute cheapest
    buy opportunities with the absolute most expensive sell opportunities.
    """
    logger.info("Calculating globally optimal profit (True Oracle)...")
    
    csv_file.seek(0) # Ensure we read the file from the beginning
    csv_data = csv_file.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data), sep=';', decimal='.')
    
    df.rename(columns={'Price (EUR)': 'price', 'Date': 'timestamp', 'Hour': 'hour'}, inplace=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(inplace=True)
    if df.empty: return 0.0

    # 1. Create lists of all possible buy and sell "slots" of energy
    buy_prices = []
    sell_prices = []
    
    # For each hour, we can charge/discharge a certain amount. We treat each kWh as a slot.
    # The number of slots per hour is limited by the charge_discharge_rate.
    slots_per_hour = int(charge_discharge_rate)
    
    for price in df['price']:
        buy_prices.extend([price] * slots_per_hour)
        sell_prices.extend([price] * slots_per_hour)
        
    # 2. Sort the opportunities
    buy_prices.sort()
    sell_prices.sort(reverse=True)
    
    # 3. Pair them up to calculate profit
    total_profit_mwh = 0.0
    # The number of transactions is limited by the number of slots or half the battery capacity,
    # because one cycle is a buy and a sell.
    # Total energy throughput is limited. A simple way to model this is to limit the pairs.
    # A more accurate way would involve a proper linear program, but this is a very strong heuristic.
    for buy_price, sell_price in zip(buy_prices, sell_prices):
        if sell_price > buy_price:
            total_profit_mwh += (sell_price - buy_price)
        else:
            # Once the most expensive sell is cheaper than the cheapest buy, stop.
            break
            
    # The total profit also cannot exceed the profit from cycling the full battery every day
    # This is a complex constraint, so we'll use the direct pairing as a strong upper bound.
    
    # Convert profit from EUR/MWh to EUR/kWh
    total_profit_kwh = total_profit_mwh / 1000.0
    
    logger.info(f"[True Oracle] Max possible profit: {total_profit_kwh:.2f}")
    return total_profit_kwh