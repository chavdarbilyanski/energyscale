import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# Assume these constants are defined elsewhere
PRICE_COLUMN = 'Price (EUR)'
HOUR_COLUMN = 'Hour'
DAY_OF_WEEK_COLUMN = 'DayOfWeek'
MONTH_COLUMN = 'Month'
PRICE_AVG24_COLUMN = 'price_rolling_avg_24h'


class BatteryEnv(gym.Env):
    """Custom Environment for a Battery Storage agent."""
    metadata = {'render_modes': ['human']}

    def __init__(self, historical_data, storage_capacity, charge_rate, efficiency):
        super(BatteryEnv, self).__init__()

        self.data = historical_data.reset_index(drop=True)
        self.storage_capacity = storage_capacity
        self.charge_rate = charge_rate
        self.efficiency = efficiency

        # ACTION SPACE: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # OBSERVATION SPACE (State)
        # We define the lower and upper bounds for each feature
        # [Price, Hour, Day, Month, Avg_Price, Battery_%]
        low_bounds = np.array(
            [0, 0, 0, 1, 0, 0], dtype=np.float32
        )
        high_bounds = np.array(
            [np.inf, 23, 6, 12, np.inf, 1.0], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # Initialize state
        self.current_step = 0
        self.current_kwh = self.storage_capacity / 2
        self.total_profit = 0

    def _get_observation(self):
        """Constructs the observation from the current state."""
        row = self.data.iloc[self.current_step]
        battery_percent = self.current_kwh / self.storage_capacity

        state = np.array([
            row[PRICE_COLUMN],
            row[HOUR_COLUMN],
            row[DAY_OF_WEEK_COLUMN],
            row[MONTH_COLUMN],
            row[PRICE_AVG24_COLUMN],
            battery_percent
        ], dtype=np.float32)
        return state

    def reset(self, seed=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_kwh = self.storage_capacity / 2
        self.total_profit = 0
        return self._get_observation(), {}

    def step(self, action):
        """Executes one time step within the environment."""
        current_row = self.data.iloc[self.current_step]
        price_per_kwh = current_row[PRICE_COLUMN] / 1000.0  # EUR/MWh to EUR/KWh

        reward = 0
        DEGRADATION_COST = 0.01 # e.g., €0.01 per cycle
        terminated = False

        # Action 1: BUY
        if action == 1 and self.current_kwh < self.storage_capacity:
            kwh_to_buy = min(self.charge_rate, self.storage_capacity - self.current_kwh)
            self.current_kwh += kwh_to_buy * self.efficiency
            reward = -(kwh_to_buy * price_per_kwh) - DEGRADATION_COST

        # Action 2: SELL
        elif action == 2 and self.current_kwh > 0:
            kwh_to_sell = min(self.charge_rate, self.current_kwh)
            self.current_kwh -= kwh_to_sell
            reward = (kwh_to_sell * price_per_kwh) - DEGRADATION_COST
            
        # Action 0: HOLD (or illegal move)
        else:
            reward = 0
            
        self.total_profit += reward

        # Move to the next time step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            terminated = True # Episode is over

        # Get the next observation
        obs = self._get_observation()
        
        # Gymnasium API returns 5 items: observation, reward, terminated, truncated, info
        return obs, reward, terminated, False, {}

    def render(self, mode='human'):
        """Renders the environment's state."""
        if mode == 'human':
            print(f"Step: {self.current_step}, "
                  f"Battery: {self.current_kwh:.2f} KWH, "
                  f"Profit: €{self.total_profit:.2f}")