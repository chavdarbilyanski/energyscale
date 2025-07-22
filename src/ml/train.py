import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces
import gymnasium as gym
import batteryEnv
import mlflow
import os

class PPOModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model"]
        stats_path = context.artifacts["stats"]
        
        # Define minimal BatteryEnv inline (no external import needed)
        class BatteryEnv(gym.Env):
            def __init__(self):
                super(BatteryEnv, self).__init__()
                self.action_space = spaces.Discrete(3)
                low_bounds = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
                high_bounds = np.array([np.inf, 23, 6, 12, np.inf, 1.0], dtype=np.float32)
                self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
            def step(self, action): return self.observation_space.sample(), 0, False, {}
            def reset(self): return self.observation_space.sample()
        
        # Use the inline class for dummy env
        vec_env = DummyVecEnv([lambda: BatteryEnv()])
        vec_env = VecNormalize.load(stats_path, vec_env)
        
        self.model = PPO.load(model_path, env=vec_env)
        self.env = vec_env  # Expose for inference
    
    def predict(self, context, model_input):
        action, _ = self.model.predict(model_input, deterministic=True)
        return action

# --- 1. Configuration ---
DATA_FILE_NAME = '/Users/chavdarbilyanski/powerbidder/src/ml/data/combine/combined_output_with_features.csv'
RL_MODEL_PATH = "battery_ppo_agent_v4 .zip"
STATS_PATH = "vec_normalize_stats_v4.pkl"
TOTAL_TIMESTEPS_MULTIPLIER = 800

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

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("Battery RL Agents")  # Add this line to use your existing experiment

with mlflow.start_run(run_name="BatteryPPO_Training_v2") as run:  # Start an MLflow run
    # Log key parameters for reproducibility
    mlflow.log_param("storage_capacity_kwh", STORAGE_CAPACITY_KWH)
    mlflow.log_param("charge_rate_kw", CHARGE_RATE_KW)
    mlflow.log_param("efficiency", EFFICIENCY)
    mlflow.log_param("total_timesteps", total_timesteps)
    mlflow.log_param("gamma", 0.999)
    mlflow.log_param("n_steps", 2048)
    mlflow.log_param("ent_coef", 0.01)
    mlflow.log_param("learning_rate", 0.0003)

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
    print("Saving and logging model artifacts to MLflow...")

    model.save(RL_MODEL_PATH)
    env.save(STATS_PATH)

    # Log as a pyfunc model with artifacts
    mlflow.pyfunc.log_model(
        artifact_path="ppo_battery_model",  # This becomes the subdir in the run
        python_model=PPOModelWrapper(),
        artifacts={
            "model": RL_MODEL_PATH,
            "stats": STATS_PATH
        },
        input_example=np.array([[0.0, 0, 0, 1, 0.0, 0.5]]),  # Sample obs to infer signature (fixes warning)
    )

    # Register using the logged model URI
    model_uri = f"runs:/{run.info.run_id}/ppo_battery_model"
    mlflow.register_model(model_uri, "BatteryPPOModel")

print(f"Training run complete. Run ID: {run.info.run_id}")