import pandas as pd
from ray.tune.logger import pretty_print
from env import PortfolioOptimizationEnv
from ray.rllib.algorithms.ppo import PPOConfig
import torch

if torch.cuda.is_available():
    print("CUDA is AVAILABLE. Number of GPUs:", torch.cuda.device_count())
else:
    print("CUDA is NOT available; using CPU only.")

### Read in data
stock_prices = pd.read_csv("train_files/stock_prices.csv")
options = pd.read_csv("train_files/options.csv")
financials = pd.read_csv("train_files/financials.csv")
secondary_stock_prices = pd.read_csv("train_files/secondary_stock_prices.csv")
trades = pd.read_csv("train_files/trades.csv")
unique_dates = stock_prices["Date"].unique()
unique_brands = stock_prices["SecuritiesCode"].unique()
### test the environment
test_stock_data = stock_prices[stock_prices["Date"].isin(unique_dates[:30])]
test_stock_data["Date"] = pd.to_datetime(test_stock_data["Date"])
test_stock_data.sort_values(by=["Date", "SecuritiesCode"], inplace=True)
test_stock_data = test_stock_data[((~test_stock_data["Open"].isnull()) 
& (~test_stock_data["High"].isnull()) 
& (~test_stock_data["Low"].isnull()) 
& (~test_stock_data["Close"].isnull()))]


config = PPOConfig().environment(
    PortfolioOptimizationEnv, env_config={
        "stock_data": test_stock_data,
        "initial_balance": 100000,
        "lookback": 5
        }
)

config.training(
        lr=0.00005,
    )
# Alternative configuration
# config.training(
#     gamma=0.99,
#     lr=0.00005,
#     kl_coeff=0.03,
#     lambda_=1,
#     clip_param=0.2,
#     num_sgd_iter=8,
#     sgd_minibatch_size=4000,
#     train_batch_size=24000,
# )

config.framework(framework = 'torch')
config.rollouts(num_rollout_workers=2)
config.resources(num_gpus=1, num_cpus_per_worker=1)

algo = config.build()
max_iter = 5

for i in range(max_iter):
        print(i)
        result = algo.train()

        if i % 100 == 0:
            print(pretty_print(result))
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")