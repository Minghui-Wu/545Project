
# Hyperparameter tuning for LightGBM using Optuna
import pandas as pd
import numpy as np
import gc
import optuna

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import TimeSeriesSplit

##############################################
# 1. Load Data
##############################################
df = pd.read_csv("training_price_features.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Date', 'SecuritiesCode']).reset_index(drop=True)
df.dropna(inplace=True)  # Clean NaNs early

drop_cols = ['Target', 'RowId']
features = [c for c in df.columns if c not in drop_cols]
X = df[features].copy()
y = df['Target'].copy()
dates = df['Date']

if 'Date' in X.columns:
    X.drop('Date', axis=1, inplace=True)

##############################################
# 2. Sharpe Function
##############################################
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    def _calc_spread_return_per_day(subdf, portfolio_size, toprank_weight_ratio):
        n_stocks = len(subdf)
        if n_stocks < portfolio_size:
            portfolio_size = n_stocks
        top_stocks = subdf.sort_values(by='Rank')['Target'][:portfolio_size]
        bottom_stocks = subdf.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size]
        weights = np.linspace(toprank_weight_ratio, 1, num=len(top_stocks))
        purchase = (top_stocks * weights).sum() / weights.mean()
        short = (bottom_stocks * weights).sum() / weights.mean()
        return purchase - short

    daily_returns = df.groupby('Date').apply(
        _calc_spread_return_per_day, portfolio_size, toprank_weight_ratio
    )
    return daily_returns.mean() / daily_returns.std()

##############################################
# 3. Time Series Split
##############################################
tscv = TimeSeriesSplit(n_splits=5)

##############################################
# 4. Optuna Objective Function
##############################################
def objective(trial):
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])

    params = {
        'boosting_type': boosting_type,
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_categorical('n_estimators', [200, 500, 1000, 2000]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': 42,
        'metric': 'mae',
        'verbosity': -1
    }

    # ✅ Conditionally disable bagging for GOSS
    if boosting_type == 'goss':
        params['subsample'] = 1.0
        params['bagging_freq'] = 0
    else:
        params['subsample'] = trial.suggest_float('subsample', 0.3, 1.0)
        params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 7)

    fold_sharpes = []

    for train_idx, val_idx in tscv.split(X, y):
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()
        date_val_fold = dates.iloc[val_idx].copy()

        # Drop NaNs in folds
        train_mask = ~X_train_fold.isnull().any(axis=1)
        X_train_fold = X_train_fold[train_mask]
        y_train_fold = y_train_fold[train_mask]

        val_mask = ~X_val_fold.isnull().any(axis=1)
        X_val_fold = X_val_fold[val_mask]
        y_val_fold = y_val_fold[val_mask]
        date_val_fold = date_val_fold[val_mask]

        model = LGBMRegressor(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='mae',
            callbacks=[early_stopping(stopping_rounds=50)]
        )

        y_pred = model.predict(X_val_fold)
        if np.isnan(y_pred).any() or np.isnan(y_val_fold).any():
            continue

        val_df = pd.DataFrame({
            'Date': date_val_fold.values,
            'Target': y_val_fold.values,
            'pred': y_pred,
            'SecuritiesCode': df['SecuritiesCode'].iloc[val_idx].values[val_mask.values]
        })
        val_df['Rank'] = val_df.groupby('Date')['pred'].rank(method='first', ascending=False) - 1

        sharpe = calc_spread_return_sharpe(val_df)
        fold_sharpes.append(sharpe)

        del X_train_fold, X_val_fold, y_train_fold, y_val_fold
        gc.collect()

    return -np.mean(fold_sharpes) if fold_sharpes else 9999

##############################################
# 5. Run Optuna Study
##############################################
sampler = TPESampler(seed=42, n_startup_trials=20)
pruner = MedianPruner(n_warmup_steps=10)

study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=300)

##############################################
# 6. Print Best Parameters
##############################################
print("\n✅ Best Value (Negative Sharpe):", study.best_value)
print("=> Best Sharpe ~", -study.best_value)
print("✅ Best Parameters:\n", study.best_params)

##############################################
# 7. Train Final Model on Full Data
##############################################
best_params = {
    **study.best_params,
    'random_state': 42,
    'metric': 'mae'
}

final_model = LGBMRegressor(**best_params)
final_model.fit(X, y)

print("\nFinal model trained on full dataset.")
