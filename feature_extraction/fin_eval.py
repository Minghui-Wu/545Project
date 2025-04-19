import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 1. Load the feature file
df1 = pd.read_csv('output_with_sentiment.csv')

# 2. Load stock price labels and ticker map
df2 = pd.read_csv('data/train_files/stock_prices.csv')      # has columns Date, SecuritiesCode, Target
df3 = pd.read_csv('data/stock_list.csv')                    # has SecuritiesCode, Name

# 3. Build reverse mapping from company name → SecuritiesCode
mapping = dict(zip(df3['SecuritiesCode'], df3['Name']))
reverse_map = {v: k for k, v in mapping.items()}
df1['SecuritiesCode'] = df1['company name'].map(reverse_map)

# 4. Standardize dates
df1['date'] = pd.to_datetime(df1['date']).dt.strftime('%Y-%m-%d')
df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%Y-%m-%d')

# 5. Merge on date + code
merged = pd.merge(
    df1,
    df2[['Date', 'SecuritiesCode', 'Target']],
    left_on=['date', 'SecuritiesCode'],
    right_on=['Date', 'SecuritiesCode'],
    how='inner'
)

print(f"Rows after join: {len(merged)}")
print("Matched companies:", merged['company name'].unique())

if merged.empty:
    print("No matches—please check date/code ranges.")
    print("Dates in df1:", df1['date'].unique())
    print("Dates in df2:", df2['Date'].unique())
    print("Codes in df1:", df1['SecuritiesCode'].dropna().unique())
    print("Codes in df2:", df2['SecuritiesCode'].unique())
else:
    # 6a. FinBERT sentiment (fin_score) vs Target
    fin_corr, fin_p = pearsonr(merged['fin_score'], merged['Target'])
    print(f"FinBERT score vs Target:     corr={fin_corr:.4f}, p-value={fin_p:.4f}")

    # 6b. Loughran–McDonald ratios vs Target
    lm_cols = ['lm_positive_ratio', 'lm_negative_ratio', 'lm_uncertainty_ratio']
    for col in lm_cols:
        corr, p = pearsonr(merged[col], merged['Target'])
        print(f"{col} vs Target: corr={corr:.4f}, p-value={p:.4f}")

    # 6c. Embedding dims vs Target
    embed_cols = [c for c in merged.columns if c.startswith('finbert_emb_')]
    embs = merged[embed_cols].values
    corrs = [pearsonr(embs[:, i], merged['Target'])[0] for i in range(embs.shape[1])]
    corrs = np.array(corrs)
    print("Embedding dims →",
          f"mean={corrs.mean():.4f}, std={corrs.std():.4f},",
          f"min={corrs.min():.4f}, max={corrs.max():.4f}")

    # 6d. Mean-embedding vs Target
    mean_emb = embs.mean(axis=1)
    m_corr, m_p = pearsonr(mean_emb, merged['Target'])
    print(f"Mean embedding vs Target: corr={m_corr:.4f}, p-value={m_p:.4f}")
