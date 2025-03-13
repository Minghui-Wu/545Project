'''
Use grid search to find the best strategy parameters based on the 4th Kaggle model
''' 

import os
from decimal import ROUND_HALF_UP, Decimal
import numpy as np
import pandas as pd
from tqdm import tqdm
from helper_files.local_api import local_api
import warnings
import concurrent.futures

base_dir = "input/jpx-tokyo-stock-exchange-prediction"

train_files_dir = f"{base_dir}/train_files"
supplemental_files_dir = f"{base_dir}/supplemental_files"

def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)

    price.set_index("Date", inplace=True)
    return price

def get_features_for_predict(price, code):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
        code (int)  : A local code for a listed company
    Returns:
        feature DataFrame (pd.DataFrame)
    """
    close_col = "AdjustedClose"
    feats = price.loc[price["SecuritiesCode"] == code, ["SecuritiesCode", close_col, "ExpectedDividend"]].copy()

    # calculate return using AdjustedClose
    feats["return_1day"] = feats[close_col].pct_change(1)
    
    # ExpectedDividend
    feats["ExpectedDividend"] = feats["ExpectedDividend"].mask(feats["ExpectedDividend"] > 0, 1)

    # filling data for nan and inf
    feats = feats.fillna(0)
    feats = feats.replace([np.inf, -np.inf], 0)
    # drop AdjustedClose column
    feats = feats.drop([close_col], axis=1)

    return feats


def run_simulation(portfolio_size, weight):
    print(f"Portfolio Size: {portfolio_size}, Weight: {weight}")
    myapi = local_api('input/jpx-tokyo-stock-exchange-prediction/train_files', 'valid', portfolio_size, weight)

    df_price_raw = pd.read_csv(f"{train_files_dir}/stock_prices.csv")
    price_cols = ["Date", "SecuritiesCode", "Close", "AdjustmentFactor", "ExpectedDividend"]
    df_price_raw = df_price_raw[price_cols]

    df_price_supplemental = pd.read_csv(f"{supplemental_files_dir}/stock_prices.csv")
    df_price_supplemental = df_price_supplemental[price_cols]
    df_price_raw = pd.concat([df_price_raw, df_price_supplemental])
    df_price_raw = df_price_raw.loc[df_price_raw["Date"] >= "2022-07-01"]

    env = myapi.make_env()
    iter_test = env.iter_test()
    counter = 0

    for (prices, options, financials, trades, secondary_prices, sample_prediction) in tqdm(iter_test):
        current_date = prices["Date"].iloc[0]
        if counter == 0:
            df_price_raw = df_price_raw.loc[df_price_raw["Date"] < current_date]
        df_price_raw = pd.concat([df_price_raw, prices[price_cols]])
        df_price = adjust_price(df_price_raw)
        codes = sorted(prices["SecuritiesCode"].unique())
        feature = pd.concat([get_features_for_predict(df_price, code) for code in codes])
        feature = feature.loc[feature.index == current_date]
        feature.loc[:, "predict"] = feature["return_1day"] + feature["ExpectedDividend"] * 100
        feature = feature.sort_values("predict", ascending=True).drop_duplicates(subset=['SecuritiesCode'])
        feature.loc[:, "Rank"] = np.arange(len(feature))
        feature_map = feature.set_index('SecuritiesCode')['Rank'].to_dict()
        sample_prediction['Rank'] = sample_prediction['SecuritiesCode'].map(feature_map)

        # Check Rank integrity
        assert sample_prediction["Rank"].notna().all()
        assert sample_prediction["Rank"].min() == 0
        assert sample_prediction["Rank"].max() == len(sample_prediction["Rank"]) - 1
        env.predict(sample_prediction)
        counter += 1
    score = env.score()

    return {"portfolio_size": portfolio_size, "weight": weight, "score": score}

if __name__ == "__main__":
    portfolio_grid = [100, 150, 200, 250]
    weight_grid = [1.5, 2.0, 2.5, 3.0]

    grid_params = [(ps, w) for ps in portfolio_grid for w in weight_grid]

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_params = {executor.submit(run_simulation, ps, w): (ps, w) for ps, w in grid_params}
        for future in concurrent.futures.as_completed(future_to_params):
            ps, w = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Portfolio Size {ps}, Weight {w} generated an exception: {exc}")

    print("All results:", results)
    results_df = pd.DataFrame(results)
    results_df.to_csv("grid_search_results.csv", index=False)