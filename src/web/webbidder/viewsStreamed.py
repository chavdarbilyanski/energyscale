import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from django.shortcuts import render
from django.conf import settings
from .forms import CSVUploadForm
import numpy as np
import gym
from gym import spaces
import logging

# Set up logging
logging.basicConfig(level=logging.INFO) # Use INFO for production to avoid overly verbose logs
logger = logging.getLogger(__name__)

# --- Constants and Environment Definition (No changes needed here) ---
MODEL_FILE_NAME = "battery_ppo_agent_v2.zip"
STATS_FILE_NAME = "vec_normalize_stats_v2.pkl"
PRICE_COLUMN = "price"
DATE_COLUMN = "timestamp"
HOUR_COLUMN = "hour"
INITIAL_BATTERY_PERCENT = 0.0

class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        low_bounds = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        high_bounds = np.array([np.inf, 23, 6, 12, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
    def step(self, action): return self.observation_space.sample(), 0, False, {}
    def reset(self): return self.observation_space.sample()

# --- Model Loading (No changes needed here) ---
def load_model_and_env():
    logger.info("Loading model and environment...")
    model_path = os.path.join(settings.BASE_DIR, 'webbidder', MODEL_FILE_NAME)
    stats_path = os.path.join(settings.BASE_DIR, 'webbidder', STATS_FILE_NAME)
    
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        error_msg = "Model or stats file not found."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    vec_env = DummyVecEnv([lambda: BatteryEnv()])
    vec_env = VecNormalize.load(stats_path, vec_env)
    model = PPO.load(model_path, env=vec_env)
    logger.info("Model and environment loaded successfully.")
    return model, vec_env

# --- REFACTORED: Main View Function ---
def upload_csv(request):
    if request.method != 'POST':
        form = CSVUploadForm()
        return render(request, 'upload.html', {'form': form})

    form = CSVUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, 'upload.html', {'form': form, 'error': f"Form invalid: {form.errors}"})

    csv_file = request.FILES['csv_file']
    max_battery_capacity = form.cleaned_data['max_battery_capacity']
    charge_discharge_rate = form.cleaned_data['charge_discharge_rate']
    
    try:
        # Load the ML model once before starting the processing loop
        model, vec_env = load_model_and_env()
        
        # --- CHUNK PROCESSING LOGIC ---
        # Initialize state variables
        battery_charge = INITIAL_BATTERY_PERCENT * max_battery_capacity
        total_profit = 0.0
        results = []
        carry_over_df = pd.DataFrame() # This will hold the last 23 rows for the rolling average
        chunk_size = 10000 # Process 10,000 rows at a time to keep memory low
        rolling_window = 24

        # Create a chunk iterator from the uploaded file
        # The 'utf-8' encoding is often needed for files from different systems
        chunk_iterator = pd.read_csv(csv_file, sep=';', decimal='.', chunksize=chunk_size, encoding='utf-8')

        for chunk in chunk_iterator:
            logger.info(f"Processing chunk of {len(chunk)} rows...")
            
            # --- 1. Preprocessing a chunk ---
            chunk.rename(columns={'Price (EUR)': PRICE_COLUMN, 'Date': DATE_COLUMN, 'Hour': HOUR_COLUMN}, inplace=True)
            required_cols = [PRICE_COLUMN, DATE_COLUMN, HOUR_COLUMN]
            if not all(col in chunk.columns for col in required_cols):
                raise ValueError("CSV chunk missing required columns.")

            chunk[PRICE_COLUMN] = pd.to_numeric(chunk[PRICE_COLUMN], errors='coerce')
            chunk[DATE_COLUMN] = pd.to_datetime(chunk[DATE_COLUMN], format='%d/%m/%Y', errors='coerce')
            chunk[HOUR_COLUMN] = pd.to_numeric(chunk[HOUR_COLUMN], errors='coerce')
            chunk.dropna(subset=required_cols, inplace=True)
            if chunk.empty:
                continue # Skip empty chunks

            chunk['DayOfWeek'] = chunk[DATE_COLUMN].dt.weekday
            chunk['Month'] = chunk[DATE_COLUMN].dt.month

            # --- 2. Handle the Rolling Average correctly ---
            # Combine the carry-over from the last chunk with the current chunk
            combined_df = pd.concat([carry_over_df, chunk], ignore_index=True)
            
            # Calculate rolling average on the combined data
            combined_df['price_rolling_avg_24h'] = combined_df[PRICE_COLUMN].rolling(window=rolling_window, min_periods=1).mean()
            
            # Get the actual rows for this chunk (without the carry-over)
            # This is the DataFrame we will iterate over for inference
            current_chunk_processed = combined_df.iloc[len(carry_over_df):].copy()
            
            # Update the carry-over for the *next* iteration
            carry_over_df = combined_df.iloc[-rolling_window:]

            # --- 3. Run Inference on the processed chunk ---
            for _, row in current_chunk_processed.iterrows():
                battery_percent = (battery_charge / max_battery_capacity) if max_battery_capacity > 0 else 0
                price_mwh = row[PRICE_COLUMN]
                price_kwh = price_mwh / 1000.0

                obs_raw = np.array([
                    price_mwh,
                    row[HOUR_COLUMN] - 1, # Model expects 0-23
                    row['DayOfWeek'],
                    row['Month'],
                    row['price_rolling_avg_24h'],
                    battery_percent
                ], dtype=np.float32)
                
                normalized_obs = vec_env.normalize_obs(np.array([obs_raw]))
                action, _ = model.predict(normalized_obs, deterministic=True)
                action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action[0], 'HOLD')
                
                # Create result dictionary
                action_details = {
                    'timestamp': row[DATE_COLUMN].strftime('%d/%m/%Y'),
                    'hour': int(row[HOUR_COLUMN]),
                    'action': action_str,
                    'price': price_kwh,
                    'battery_charge': battery_charge,
                    'battery_percent': battery_percent * 100.0,
                    'profit_change': 0.0,
                    'total_profit': total_profit
                }
                
                # Apply action
                if action_str == 'BUY' and battery_charge < max_battery_capacity:
                    amount = min(charge_discharge_rate, max_battery_capacity - battery_charge)
                    action_details['profit_change'] = -price_kwh * amount
                    total_profit += action_details['profit_change']
                    battery_charge += amount
                elif action_str == 'SELL' and battery_charge > 0:
                    amount = min(charge_discharge_rate, battery_charge)
                    action_details['profit_change'] = price_kwh * amount
                    total_profit += action_details['profit_change']
                    battery_charge -= amount

                results.append(action_details)
        
        # --- End of loop ---
        logger.info(f"Finished processing all chunks. Total rows: {len(results)}, Total profit: {total_profit:.2f}")

        return render(request, 'results.html', {
            'battery_charge': battery_charge,
            'battery_percent': (battery_charge / max_battery_capacity) * 100.0 if max_battery_capacity > 0 else 0,
            'total_profit': total_profit,
            'results': results,
            'max_battery_capacity': max_battery_capacity,
            'charge_discharge_rate': charge_discharge_rate
        })
    except Exception as e:
        logger.exception("An error occurred during file processing.") # Logs the full traceback
        return render(request, 'upload.html', {
            'form': form,
            'error': f"An unexpected error occurred: {str(e)}"
        })