import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_PATH = "battery_ppo_agent_v2.zip"
STATS_PATH = "vec_normalize_stats_v2.pkl"
HISTORICAL_DATA_FILE = '/Users/chavdarbilyanski/powerbidder/combine/combined_output_with_features.csv'

# --- Simulation Parameters ---
STORAGE_CAPACITY_KWH = 100.0  # Your battery's capacity
CHARGE_RATE_KW = 25.0         # Max charge/discharge rate per hour (in kW)
EFFICIENCY = 0.90             # Round-trip efficiency (90%)
INITIAL_KWH = STORAGE_CAPACITY_KWH / 2 # Start with a half-full battery

# Column names
PRICE_COLUMN = 'Price (EUR)'
HOUR_COLUMN = 'Hour'
DAY_OF_WEEK_COLUMN = 'DayOfWeek'
MONTH_COLUMN = 'Month'
PRICE_AVG24_COLUMN = 'price_rolling_avg_24h'
DATE_COLUMN = 'Date' # Make sure your CSV has a 'Date' column for the index

# --- ENVIRONMENT DEFINITION (Minimal version for loading stats) ---
class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        low_bounds = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        high_bounds = np.array([np.inf, 23, 6, 12, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
    def step(self, action): return self.observation_space.sample(), 0, False, False, {}
    def reset(self, seed=None): return self.observation_space.sample(), {}


# --- MAIN BACKTESTING LOGIC ---
if __name__ == "__main__":
    # 1. Load and Prepare Historical Data
    print("Loading historical data...")
    df = pd.read_csv(HISTORICAL_DATA_FILE, sep=';', decimal=',')
    df.rename(columns={'Price (EUR)': PRICE_COLUMN, 'Hour': HOUR_COLUMN, 
                       'DayOfWeek': DAY_OF_WEEK_COLUMN, 'Month': MONTH_COLUMN, 
                       'Date': DATE_COLUMN}, inplace=True)
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors='coerce')
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format='%m/%d/%y', errors='coerce')
    df.dropna(inplace=True)
    df.set_index(DATE_COLUMN, inplace=True)
    df[PRICE_AVG24_COLUMN] = df[PRICE_COLUMN].rolling(window=24, min_periods=1).mean()
    print(f"Data loaded. Shape: {df.shape}")

    # 2. Load the RL Model and Normalization Statistics
    print("Loading model and normalization stats...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
        raise FileNotFoundError("Model or stats file not found. Ensure both files are in the directory.")
        
    vec_env = DummyVecEnv([lambda: BatteryEnv()])
    vec_env = VecNormalize.load(STATS_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    model = PPO.load(MODEL_PATH, env=vec_env)
    print("Model and stats loaded successfully.")

    # 3. Initialize Simulation State
    print("Initializing backtest...")
    current_kwh = INITIAL_KWH
    total_profit = 0.0
    action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    log = []

    # 4. Run the Backtest Loop
    print(f"Running simulation over {len(df)} time steps...")
    for index, row in df.iterrows():
        # A. Construct the current observation
        current_storage_percent = current_kwh / STORAGE_CAPACITY_KWH
        
        obs_raw = np.array([
            row[PRICE_COLUMN],
            row[HOUR_COLUMN],
            row[DAY_OF_WEEK_COLUMN],
            row[MONTH_COLUMN],
            row[PRICE_AVG24_COLUMN],
            current_storage_percent
        ], dtype=np.float32)

        # B. Normalize the observation and predict action
        obs_normalized = vec_env.normalize_obs(np.array([obs_raw]))
        action_code, _ = model.predict(obs_normalized, deterministic=True)
        action_label = action_map[action_code[0]]
        
        # C. Execute the trade and update state
        price_per_kwh = row[PRICE_COLUMN] / 1000.0  # Convert EUR/MWh to EUR/KWh
        kwh_traded = 0
        trade_profit = 0

        if action_label == "BUY" and current_kwh < STORAGE_CAPACITY_KWH:
            kwh_to_buy = min(CHARGE_RATE_KW, STORAGE_CAPACITY_KWH - current_kwh)
            current_kwh += kwh_to_buy * EFFICIENCY
            trade_profit = -(kwh_to_buy * price_per_kwh)
            kwh_traded = kwh_to_buy
        
        elif action_label == "SELL" and current_kwh > 0:
            kwh_to_sell = min(CHARGE_RATE_KW, current_kwh)
            current_kwh -= kwh_to_sell
            trade_profit = kwh_to_sell * price_per_kwh
            kwh_traded = -kwh_to_sell # Negative to indicate discharge

        total_profit += trade_profit
        
        # D. Log the transaction for later analysis
        log.append({
            'timestamp': index,
            'price_eur_mwh': row[PRICE_COLUMN],
            'battery_kwh': current_kwh,
            'battery_percent': current_storage_percent,
            'action': action_label,
            'kwh_traded': kwh_traded,
            'profit': trade_profit,
            'cumulative_profit': total_profit
        })

    print("Backtest complete.")

    # 5. Analyze and Report Results
    log_df = pd.DataFrame(log)
    log_df.set_index('timestamp', inplace=True)
    
    print("\n--- Backtest Results ---")
    print(f"Total Cumulative Profit: €{log_df['cumulative_profit'].iloc[-1]:.2f}")
    
    buy_trades = log_df[log_df['action'] == 'BUY']
    sell_trades = log_df[log_df['action'] == 'SELL']
    
    print(f"\nTotal Buy Trades: {len(buy_trades)}")
    print(f"Total Sell Trades: {len(sell_trades)}")
    print(f"Average Profit per Sell Trade: €{sell_trades['profit'].mean():.4f}")
    
    # 6. Visualize the Results
    print("Generating performance plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Price and Trades
    ax1.plot(log_df.index, log_df['price_eur_mwh'], label='Price (€/MWh)', color='gray', alpha=0.6)
    ax1.scatter(buy_trades.index, buy_trades['price_eur_mwh'], color='green', label='Buy', marker='^', s=50, alpha=0.8)
    ax1.scatter(sell_trades.index, sell_trades['price_eur_mwh'], color='red', label='Sell', marker='v', s=50, alpha=0.8)
    ax1.set_ylabel('Price (€/MWh)')
    ax1.set_title('Electricity Price and Agent Actions')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Battery Charge and Cumulative Profit
    ax2_profit = ax1.twinx() # Create a second y-axis for profit
    ax2.plot(log_df.index, log_df['battery_kwh'], label='Battery Level (KWH)', color='blue')
    ax2_profit.plot(log_df.index, log_df['cumulative_profit'], label='Cumulative Profit (€)', color='orange', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Battery Level (KWH)', color='blue')
    ax2_profit.set_ylabel('Cumulative Profit (€)', color='orange')
    ax2.set_title('Battery State and Cumulative Profit')
    ax2.legend(loc='upper left')
    ax2_profit.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_performance.png')
    print("Plot saved to backtest_performance.png")
    plt.show()