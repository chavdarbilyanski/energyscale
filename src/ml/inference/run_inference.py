import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import os

# --- CONFIGURATION ---
MODEL_PATH = "battery_ppo_agent.zip"
STATS_PATH = "vec_normalize_stats.pkl"

# --- ENVIRONMENT DEFINITION ---
class BatteryEnv(gym.Env):
    """Minimal environment definition needed for structure."""
    def __init__(self):
        super(BatteryEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        low_bounds = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        high_bounds = np.array([np.inf, 23, 6, 12, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
    
    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}
    
    def reset(self, seed=None):
        return self.observation_space.sample(), {}

# --- HELPER FUNCTIONS ---
def get_live_data():
    print("Fetching live market data...")
    return {
        'Price (EUR)': 65.50,
        'price_rolling_avg_24h': 95.20,
        'current_time': datetime.now()
    }

def get_battery_state():
    print("Fetching live battery state...")
    return 0.3

def execute_action(action_label):
    print("="*30)
    print(f"  ACTION EXECUTED: {action_label}")
    print("="*30)

# --- MAIN INFERENCE LOGIC (FINAL ROBUST VERSION) ---
if __name__ == "__main__":
    try:
        print(f"--- Starting Inference Run at {datetime.now()} ---")
        
        # 1. Create a dummy VecEnv to load the normalization stats into.
        # This does NOT need to be passed to the model load function.
        if not os.path.exists(STATS_PATH):
            raise FileNotFoundError(f"Error: Statistics file not found at {STATS_PATH}.")
        
        vec_env = DummyVecEnv([lambda: BatteryEnv()])
        vec_env = VecNormalize.load(STATS_PATH, vec_env)
        print(f"Normalization stats from {STATS_PATH} loaded successfully.")

        # 2. Load the model. We don't need to pass the env here.
        model = PPO.load(MODEL_PATH)
        print(f"Model {MODEL_PATH} loaded successfully.")
        
        # 3. Get live data
        market_data = get_live_data()
        battery_percent = get_battery_state()
        
        # 4. Construct the raw observation array
        now = market_data['current_time']
        obs_raw = np.array([
            market_data['Price (EUR)'],
            now.hour,
            now.weekday(),
            now.month,
            market_data['price_rolling_avg_24h'],
            battery_percent
        ], dtype=np.float32)
        
        print(f"Constructed Observation (raw): {obs_raw}")

        # 5. *** THE CRUCIAL FIX ***
        # Manually normalize the observation using the loaded stats from the environment.
        # The `normalize_obs` method expects a 2D array, so we wrap `obs_raw`.
        normalized_obs = vec_env.normalize_obs(np.array([obs_raw]))
        print(f"Observation after normalization: {normalized_obs}")
        
        # 6. Predict using the NORMALIZED observation.
        action, _ = model.predict(normalized_obs, deterministic=True)
        
        # 7. Map and Execute
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        # The action is returned as an array, so access the first element
        action_label = action_map[action[0]]
        
        execute_action(action_label)

    except Exception as e:
        print(f"An error occurred during inference: {e}")
