import pandas as pd
from ray.tune.logger import pretty_print
from trading_env import PortfolioOptimizationEnv_with_feature
from ray.rllib.algorithms.ppo import PPOConfig
import torch

def train():
    if torch.cuda.is_available():
        print("CUDA is AVAILABLE. Number of GPUs:", torch.cuda.device_count())
    else:
        print("CUDA is NOT available; using CPU only.")

    ### Read in data
    stock_prices = pd.read_csv("train_files/stock_prices.csv")
    unique_dates = stock_prices["Date"].unique()
    unique_brands = stock_prices["SecuritiesCode"].unique()
    features = pd.read_csv("rl_features.csv")

    config = PPOConfig().environment(
        PortfolioOptimizationEnv_with_feature, env_config={
            "stock_data": features,
            "initial_balance": 100000,
            "lookback": 1
            }
    )

    config.training(
            lr=0.00005,
        )

    config.framework(framework = 'torch')
    config.rollouts(num_rollout_workers=30)
    config.resources(num_gpus=1, num_cpus_per_worker=1)

    algo = config.build()
    max_iter = 2000000
    checkpoint_dir_path = "./trading2"
    for i in range(max_iter):
            print(i)
            result = algo.train()

            if i % 50 == 0:
                print(pretty_print(result))
                checkpoint_dir = algo.save()
                print(f"Checkpoint saved in directory {checkpoint_dir}")

if __name__ == "__main__":
     train()