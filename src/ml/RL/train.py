import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import batteryEnv

# --- 1. Configuration ---
DATA_FILE_NAME = '/Users/chavdarbilyanski/powerbidder/src/ml/data/combine/combined_output_with_features.csv'
RL_MODEL_PATH = "battery_ppo_agent_v2.zip"
STATS_PATH = "vec_normalize_stats_v2.pkl"
TOTAL_TIMESTEPS_MULTIPLIER = 400

# Column names
DATE_COLUMN = 'Date'
PRICE_COLUMN = 'Price (EUR)'
# ... other column names if needed by your env ...

# --- 2. Data Loading and Preparation ---
print("Loading and preparing historical data...")
# Load data into a DataFrame named 'dataset'
dataset = pd.read_csv(DATA_FILE_NAME, sep=';', decimal='.') # Corrected decimal separator to comma
dataset.rename(columns={'Price (EUR)': PRICE_COLUMN, 'Date': DATE_COLUMN}, inplace=True)
# Add any other data cleaning or feature engineering here...
dataset[PRICE_COLUMN] = pd.to_numeric(dataset[PRICE_COLUMN], errors='coerce')
dataset[DATE_COLUMN] = pd.to_datetime(dataset[DATE_COLUMN], format='%m/%d/%y', errors='coerce')
dataset.dropna(inplace=True)
dataset['price_rolling_avg_24h'] = dataset[PRICE_COLUMN].rolling(window=24, min_periods=1).mean()
dataset.set_index(DATE_COLUMN, inplace=True)
dataset.sort_index(inplace=True)
print(f"Data loaded and processed. Shape: {dataset.shape}")

# Define the parameters that your BatteryEnv needs
STORAGE_CAPACITY_KWH = 100.0
CHARGE_RATE_KW = 25.0
EFFICIENCY = 0.90
# --- 3. Create and Wrap the Environment ---
print("Creating and wrapping the environment...")
env_creator = lambda: batteryEnv.BatteryEnv(
    historical_data=dataset,
    storage_capacity=STORAGE_CAPACITY_KWH,
    charge_rate=CHARGE_RATE_KW,
    efficiency=EFFICIENCY
)

env = DummyVecEnv([env_creator])
env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.999)

print("Environment created successfully.")

# --- 4. Define and Train the Model ---
total_timesteps = len(dataset) * TOTAL_TIMESTEPS_MULTIPLIER

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ppo_battery_tensorboard/",
    gamma=0.999,
    n_steps=2048,
    ent_coef=0.01,
    learning_rate=0.0003
)

print(f"--- Starting new training run for {total_timesteps:,} timesteps ---")
model.learn(total_timesteps=total_timesteps, progress_bar=True)
print("--- Training complete ---")

# --- 5. Save the Model and Normalization Stats ---
print("Saving model and normalization stats...")
model.save(RL_MODEL_PATH)
env.save(STATS_PATH) # This now correctly calls .save() on the VecNormalize object

print(f"Model saved to: {RL_MODEL_PATH}")
print(f"Normalization stats saved to: {STATS_PATH}")