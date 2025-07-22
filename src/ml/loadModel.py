from stable_baselines3 import PPO
import numpy as np

# Assume BatteryEnv class is defined in this script or imported
# You still need the environment definition to run the model
# env = BatteryEnv(...) 

# --- Load the saved RL agent ---
print("--- Loading RL agent from disk ---")
RL_MODEL_PATH = "battery_ppo_agent.zip"
loaded_rl_model = PPO.load(RL_MODEL_PATH)
print("RL Agent loaded successfully.")


# --- Prepare a new observation (state) ---
# NOTE: RL models predict on a single observation (NumPy array), not a DataFrame
# The order and data types must match the observation_space EXACTLY.
# [Price, Hour, Day, Month, Avg_Price, Battery_%]
new_observation = np.array([
    25.0,     # Price
    2,        # Hour
    1,        # Day of Week
    7,        # Month
    105.0,    # 24h Avg Price
    0.5       # Battery %
], dtype=np.float32)


# --- Use the loaded RL model to predict an action ---
# The predict method handles normalization automatically!
# deterministic=True ensures it always picks the best action, not a random one for exploration.
action, _states = loaded_rl_model.predict(new_observation, deterministic=True)

# Map the numeric action back to a meaningful label
action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
print(f"\nNew Observation: {new_observation}")
print(f"Predicted Action: >>> {action_map[action]} <<<")