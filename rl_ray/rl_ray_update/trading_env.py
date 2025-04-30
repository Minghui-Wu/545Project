import numpy as np 
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces

class PortfolioOptimizationEnv_with_feature(Env):
    print("updated")
    def __init__(self, env_config):
        """
        Portfolio Optimization Environment with efficient indexing using dictionaries.
        
        :param stock_data: DataFrame containing stock data.
        :param initial_balance: Starting capital for the agent.
        :param lookback: Number of past days included in the state observation.
        """
        super(PortfolioOptimizationEnv_with_feature, self).__init__()

        self.stock_data = env_config["stock_data"]
        self.initial_balance = env_config.get("initial_balance", 100000)
        self.lookback = env_config.get("lookback", 1)

        # Extract unique dates and securities
        self.unique_dates = sorted(self.stock_data["Date"].unique())  # Sorted unique dates
        self.unique_stocks = sorted(self.stock_data["SecuritiesCode"].unique())  # Sorted unique securities
        self.num_assets = len(self.unique_stocks)

        # Create a dictionary to map dates to index positions
        self.date_to_index = {date: idx for idx, date in enumerate(self.unique_dates)}

        # Create a dictionary to map securities to index positions
        self.stock_to_index = {stock: idx for idx, stock in enumerate(self.unique_stocks)}

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback* self.num_assets*1,), dtype=np.float64
        )

        # Define action space (allocation weights per stock)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float64)

        # Initial state
        self.current_step = self.lookback + 1
        self.done = False
        self.cash_balance = self.initial_balance
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets

    def step(self, action):
        """
        Executes a trade action and calculates the reward.
        """
        prev_step = self.current_step
        self.current_step += 1 

        # Normalize action weights to ensure sum = 1
        action = action / np.sum(action)
        prev_weights = self.portfolio_weights
        self.portfolio_weights = action
        
        if self.current_step >= list(self.date_to_index.values())[-1]:
            self.done = True
            return self._get_observation(), 0, self.done, False, {}
        
        # Stock data of previous and current time step
        stock_subset_prev = self.stock_data[self.stock_data['Date'].apply(lambda x : self.date_to_index[x]) == prev_step]
        stock_subset = self.stock_data[self.stock_data['Date'].apply(lambda x : self.date_to_index[x]) == self.current_step]
        # Filter out brand that doesn't have close price for the current time step
       # Ensure full stock universe size (1864 stocks)
        full_stock_prices_prev = np.zeros(self.num_assets)  # Fill missing with 0
        full_stock_prices_curr = np.zeros(self.num_assets)  # Fill missing with 0
        
        # Create a dictionary mapping stock codes to close prices for quick lookup
        prev_price_dict = dict(zip(stock_subset_prev["SecuritiesCode"], stock_subset_prev["Close"]))
        curr_price_dict = dict(zip(stock_subset["SecuritiesCode"], stock_subset["Close"]))
        
        # Map stock prices to their correct index positions
        for i, stock in enumerate(self.unique_stocks):  # unique_stocks contains all 1864 stock codes
            if stock in prev_price_dict:
                full_stock_prices_prev[i] = prev_price_dict[stock]
            if stock in curr_price_dict:
                full_stock_prices_curr[i] = curr_price_dict[stock]
        
        # Now, dot product will always work since both arrays have the same length
        prev_portfolio_value = np.dot(prev_weights, full_stock_prices_prev)
        new_portfolio_value = np.dot(self.portfolio_weights, full_stock_prices_curr)
        # Compute final reward as portfolio value difference
        reward = new_portfolio_value - prev_portfolio_value

        # Update cash balance to new portfolio value
        self.cash_balance = new_portfolio_value

        return self._get_observation(), reward, self.done, False, {}

    def reset(self, *, seed = None, options = None):
        """
        Resets the environment to the initial state.
        """
        super().reset(seed=seed)
        
        self.current_step = self.lookback + 1
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets  # Equal allocation
        self.cash_balance = self.initial_balance
        self.done = False
        return self._get_observation(), {}

    def _get_observation(self):
        """
        Fetches historical stock data without pivot and avoids index errors.
        Ensures all securities are included and fills missing values appropriately.
        """
        # Ensure lookback window stays within valid range
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = min(self.current_step, len(self.unique_dates) - 1)
    
        # Get start and end dates
        start_date = self.unique_dates[start_idx]
        end_date = self.unique_dates[end_idx]
    
        # Fetch stock data for the given period
        obs_df = self.stock_data[
            (self.stock_data["Date"] >= start_date) & 
            (self.stock_data["Date"] <= end_date)
        ]
    
        # Ensure data is sorted properly
        obs_df = obs_df.sort_values(by=["Date", "SecuritiesCode"])
    
        # Initialize observation array
        obs_array = np.full((self.lookback, self.num_assets, 5), np.nan)  # Default to NaN for proper filling
    
        # Fill the observation array with stock data
        for i, date in enumerate(self.unique_dates[start_idx:end_idx]):
            daily_data = obs_df[obs_df["Date"] == date]
    
            # Convert daily_data to a dictionary for fast lookups
            stock_dict = daily_data.set_index("SecuritiesCode")[["AdjustedClose", "ExpectedDividend","MovingAvg_5Day", "ExpMovingAvg_5Day", "Volatility_5Day"]].to_dict(orient="index")
    
            for j, stock in enumerate(self.unique_stocks):
                if stock in stock_dict:
                    obs_array[i, j, 0] = stock_dict[stock]["AdjustedClose"]
                    obs_array[i, j, 1] = stock_dict[stock]["ExpectedDividend"]
                    obs_array[i, j, 2] = stock_dict[stock]["MovingAvg_5Day"]
                    obs_array[i, j, 3] = stock_dict[stock]["ExpMovingAvg_5Day"]
                    obs_array[i, j, 4] = stock_dict[stock]["Volatility_5Day"]
    
        # Fill missing values using forward fill along the time axis
        obs_array = pd.DataFrame(obs_array.reshape(-1, 1)).fillna(method="bfill").values.reshape(self.lookback, self.num_assets, 5)
    
        # If any remaining NaNs exist, replace with 0
        obs_array = np.nan_to_num(obs_array)
    
        return obs_array.reshape(-1)