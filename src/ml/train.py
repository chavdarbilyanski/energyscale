import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from gymnasium import spaces
import gymnasium as gym
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import torch
import batteryEnv
import mlflow
import os
from typing import Callable
# Optional: Checkpoint callback for long trainings (save every 10k steps, useful for GCP interruptions)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="ppo_bess")

class PPOModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model"]
        stats_path = context.artifacts["stats"]
        
        # Create dummy data and params for env instantiation (only for spaces; not used in predict)
        dummy_data = pd.DataFrame({
            PRICE_COLUMN: [0.0],
            HOUR_SIN: [0],
            HOUR_COS: [0],
            MONTH_SIN: [0],
            MONTH_COS: [0],
            PRICE_AVG24_COLUMN: [0.0]
        })
        dummy_capacity = 1.0
        dummy_rate = 1.0
        dummy_efficiency = 0.9
        
        vec_env = DummyVecEnv([lambda: batteryEnv.BatteryEnv(
            historical_data=dummy_data,
            storage_capacity=dummy_capacity,
            charge_rate=dummy_rate,
            efficiency=dummy_efficiency
        )])
        vec_env = VecNormalize.load(stats_path, vec_env)
        
        self.model = PPO.load(model_path, env=vec_env)
        self.env = vec_env  # Expose for inference (normalization)
    
    def predict(self, context, model_input):
        # model_input: raw obs as np.array (n_envs, obs_dim), e.g., (1, 6)
        normalized_input = self.env.normalize_obs(model_input)
        action, _ = self.model.predict(normalized_input, deterministic=True)
        return action

# This is a helper function to create a linear scheduler
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        """
        return progress_remaining * initial_value

    return func

# --- 1. Configuration ---
DATA_FILE_NAME = '/Users/chavdarbilyanski/powerbidder/src/ml/data/combine/combined_output_with_features.csv'
RL_MODEL_PATH = "../models/PPO_Cyclical.zip"
STATS_PATH = "../models/PPO_Cycli_vec_normalize_stats.pkl"
TOTAL_TIMESTEPS_MULTIPLIER = 40

# Column names
DATE_COLUMN = 'Date'
PRICE_COLUMN = 'Price (EUR)'
HOUR_SIN = 'Hour Sin'
HOUR_COS = 'Hour Cos'
MONTH_SIN = 'Mont Sin'
MONTH_COS = 'Month Cos'
PRICE_AVG24_COLUMN = 'price_rolling_avg_24h'
# ... other column names if needed by your env ...

# --- 2. Data Loading and Preparation ---
print("Loading and preparing historical data...")
# Load data into a DataFrame named 'dataset'
dataset = pd.read_csv(DATA_FILE_NAME, sep=';', decimal=',') 
dataset.rename(columns={'Price (EUR)': PRICE_COLUMN, 'Date': DATE_COLUMN}, inplace=True)
# Add any other data cleaning or feature engineering here...
dataset[PRICE_COLUMN] = pd.to_numeric(dataset[PRICE_COLUMN], errors='coerce')
dataset[DATE_COLUMN] = pd.to_datetime(dataset[DATE_COLUMN], format='%m/%d/%y', errors='coerce')
dataset.dropna(inplace=True)
dataset[PRICE_AVG24_COLUMN] = dataset[PRICE_COLUMN].rolling(window=24, min_periods=1).mean()
dataset.set_index(DATE_COLUMN, inplace=True)
dataset.sort_index(inplace=True)
print(f"Data loaded and processed. Shape: {dataset.shape}")

# Make all temporal features cyclical
dataset['hour_sin'] = np.sin(2 * np.pi * dataset['Hour'] / 24)
dataset['hour_cos'] = np.cos(2 * np.pi * dataset['Hour'] / 24)
dataset['dayofweek_sin'] = np.sin(2 * np.pi * dataset['DayOfWeek'] / 6) # It is 0-6
dataset['dayofweek_cos'] = np.cos(2 * np.pi * dataset['DayOfWeek'] / 6) # It is 0-6
dataset['month_sin'] = np.sin(2 * np.pi * dataset['Month'] / 12)
dataset['month_cos'] = np.cos(2 * np.pi * dataset['Month'] / 12)

# Simple LSTM forecaster (log to MLflow)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
lstm_uri = "models:/BESS_Price_Forecaster/Latest"
lstm_wrapper = mlflow.pyfunc.load_model(lstm_uri)

# Precompute forecasted_price_mean (sliding window)
forecasts = []
seq_len = 24
for i in range(len(dataset) - seq_len):
    seq = dataset['Price (EUR)'].values[i:i+seq_len].reshape(1, seq_len, 1)
    forecast_mean = lstm_wrapper.predict(seq)[0][0]  # Mean of next 24h
    forecasts.append(forecast_mean)
forecasts = [0] * seq_len + forecasts  # Pad beginning
dataset['forecasted_price_mean'] = forecasts

# Define the parameters that your BatteryEnv needs
STORAGE_CAPACITY_KWH = 215.0
CHARGE_RATE_KW = 40.0
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

mlflow.set_experiment("Battery RL Agents")  # Add this line to use your existing experiment

with mlflow.start_run(run_name="Ciclical Time") as run:  # Start an MLflow run
    

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./ppo_battery_tensorboard/",
        gamma=0.99999,
        n_steps=2048,
        ent_coef=0.01,
        learning_rate=linear_schedule(0.0003)
    )

    # Log key parameters for reproducibility
    mlflow.log_param("storage_capacity_kwh", STORAGE_CAPACITY_KWH)
    mlflow.log_param("charge_rate_kw", CHARGE_RATE_KW)
    mlflow.log_param("efficiency", EFFICIENCY)
    mlflow.log_param("total_timesteps", total_timesteps)
    mlflow.log_param("gamma", model.gamma)
    mlflow.log_param("n_steps", model.n_steps)
    mlflow.log_param("ent_coef", model.ent_coef)
    mlflow.log_param("learning_rate", model.learning_rate)


    print(f"--- Starting new training run for {total_timesteps:,} timesteps ---")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)
    print("--- Training complete ---")
    # --- 5. Save and Log the Model and Normalization Stats ---
    print("Saving and logging model artifacts to MLflow...")
    # Ensure save paths exist
    os.makedirs(os.path.dirname(RL_MODEL_PATH), exist_ok=True)
    model.save(RL_MODEL_PATH)
    env.save(STATS_PATH)

    # Log the custom pyfunc model (wraps PPO and VecNormalize for easy loading/inference)
    artifacts = {
        "model": RL_MODEL_PATH,
        "stats": STATS_PATH
    }
    mlflow.pyfunc.log_model(
        artifact_path="model",  # Standard path; change if needed, but keep consistent
        python_model=PPOModelWrapper(),
        artifacts=artifacts,
        code_paths=[os.path.join(os.path.dirname(__file__), "batteryEnv.py")]  # Relative to train.py dir       
    )

    # Register the model in MLflow Model Registry
    model_uri = f"runs:/{run.info.run_id}/model"
    registered_model = mlflow.register_model(model_uri, "BatteryPPOModel")
    print(f"Model registered as version {registered_model.version} of 'BatteryPPOModel'.")

    print(f"Training run complete. Run ID: {run.info.run_id}")