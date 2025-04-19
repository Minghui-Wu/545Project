import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load enriched features
df1 = pd.read_csv('output_with_sentiment.csv')

# 2. Load labels and ticker map
df2 = pd.read_csv('data/train_files/stock_prices.csv')  # Date, SecuritiesCode, Target
df3 = pd.read_csv('data/stock_list.csv')                # SecuritiesCode, Name

# 3. Map company_name → SecuritiesCode
mapping = dict(zip(df3['SecuritiesCode'], df3['Name']))
reverse_map = {v: k for k, v in mapping.items()}
df1['SecuritiesCode'] = df1['company name'].map(reverse_map)

# 4. Align dates
df1['date'] = pd.to_datetime(df1['date']).dt.strftime('%Y-%m-%d')
df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%Y-%m-%d')

# 5. Merge on date & code
merged = pd.merge(
    df1,
    df2[['Date', 'SecuritiesCode', 'Target']],
    left_on=['date', 'SecuritiesCode'],
    right_on=['Date', 'SecuritiesCode'],
    how='inner'
)

# 6. Prepare target and feature matrices
y = merged['Target'].values

# Baseline: intercept-only
X_base = np.ones((len(merged), 1))

# Augmented: select text features + mean embedding
# Ensure embedding columns exist
embed_cols = [c for c in merged.columns if c.startswith('finbert_emb_')]
merged['mean_emb'] = merged[embed_cols].mean(axis=1)

feature_cols = ['fin_score', 'mean_emb']
X_aug = merged[feature_cols].values

# 7. TimeSeries split for evaluation
tscv = TimeSeriesSplit(n_splits=5)

results = {}
for name, X in [('Baseline', X_base), ('Text-Augmented', X_aug)]:
    mse_scores = []
    r2_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    results[name] = {
        'MSE_mean': np.mean(mse_scores), 'MSE_std': np.std(mse_scores),
        'R2_mean': np.mean(r2_scores), 'R2_std': np.std(r2_scores)
    }

# 8. Display results
res_df = pd.DataFrame(results).T
print("Evaluation Metrics (TimeSeriesSplit, 5 folds):")
print(res_df)
