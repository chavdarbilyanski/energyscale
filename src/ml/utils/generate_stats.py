# File: generate_stats.py
# Purpose: To create the VecNormalize stats file that corresponds to an existing model.

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces
import gymnasium as gym
import numpy as np

# --- Re-create your environment class exactly as it was during training ---
# (Copy this from your training script)
# --- CONFIGURATION ---
DATA_FILE_NAME = '/Users/chavdarbilyanski/powerbidder/combine/combined_output_with_features.csv'
PRICE_COLUMN = 'Price (EUR)'
HOUR_COLUMN = 'Hour'
DAY_OF_WEEK_COLUMN = 'DayOfWeek'
MONTH_COLUMN = 'Month'
PRICE_AVG24_COLUMN = 'price_rolling_avg_24h'
STORAGE_CURRENT_PERCENT = 'current_storage_percent'

class BatteryEnv(gym.Env):
    def __init__(self, historical_data, storage_capacity, charge_rate, efficiency):
        super(BatteryEnv, self).__init__()
        self.data = historical_data.reset_index(drop=True)
        self.storage_capacity = storage_capacity
        # ... (rest of your init)

        self.action_space = spaces.Discrete(3)
        low_bounds = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        high_bounds = np.array([np.inf, 23, 6, 12, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        
        # Initialize state for a few steps
        self.current_step = 0
        self.current_kwh = self.storage_capacity / 2
        
    def reset(self, seed=None):
        self.current_step = 0
        self.current_kwh = self.storage_capacity / 2
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        battery_percent = self.current_kwh / self.storage_capacity
        state = np.array([ row[PRICE_COLUMN], row[HOUR_COLUMN], row[DAY_OF_WEEK_COLUMN],
                           row[MONTH_COLUMN], row[PRICE_AVG24_COLUMN], battery_percent
        ], dtype=np.float32)
        return state

    def step(self, action):
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        # The reward logic doesn't matter here, we just need the structure
        return self._get_observation(), 0, terminated, False, {}

# --- SCRIPT LOGIC ---

# 1. Load your data (needed to initialize the env)
dataset = pd.read_csv(DATA_FILE_NAME, sep=';', decimal='.')
# (Add any other preprocessing you did before training)
dataset.rename(columns={'Price (EUR)': PRICE_COLUMN, 'Hour': HOUR_COLUMN, 'DayOfWeek': DAY_OF_WEEK_COLUMN, 'Month': MONTH_COLUMN}, inplace=True)
dataset[PRICE_COLUMN] = pd.to_numeric(dataset[PRICE_COLUMN], errors='coerce')
dataset['Volume'] = pd.to_numeric(dataset['Volume'], errors='coerce')
dataset.dropna(inplace=True)
dataset['price_rolling_avg_24h'] = dataset[PRICE_COLUMN].rolling(window=24, min_periods=1).mean()


# 2. Create and wrap the environment
# This MUST be the same wrapping process as your original training and inference
env = DummyVecEnv([lambda: BatteryEnv(dataset, 100.0, 25, 0.9)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 3. Train for a tiny amount of time (e.g., 1000 steps)
# The goal is not to learn a new policy, but to make the VecNormalize wrapper
# calculate and store the running mean and standard deviation of your data.
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1000)

# 4. Save the crucial statistics file
STATS_PATH = "vec_normalize_stats.pkl"
env.save(STATS_PATH)

print(f"Successfully generated and saved normalization stats to {STATS_PATH}")
print("You can now upload this file along with your original battery_ppo_agent.zip")