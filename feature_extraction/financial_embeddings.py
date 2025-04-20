import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 1. Load enriched feature file
df1 = pd.read_csv('output_with_sentiment.csv')

# 2. Load price labels and ticker map
df2 = pd.read_csv('data/train_files/stock_prices.csv')   # cols: Date, SecuritiesCode, Target
df3 = pd.read_csv('data/stock_list.csv')                 # cols: SecuritiesCode, Name

# 3. Map company_name → SecuritiesCode
reverse_map = {v: k for k, v in zip(df3['SecuritiesCode'], df3['Name'])}
df1['SecuritiesCode'] = df1['company name'].map(reverse_map)

# 4. Standardize dates
df1['date'] = pd.to_datetime(df1['date']).dt.strftime('%Y-%m-%d')
df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%Y-%m-%d')

# 5. Merge on date + code
merged = pd.merge(
    df1,
    df2[['Date','SecuritiesCode','Target']],
    left_on=['date','SecuritiesCode'],
    right_on=['Date','SecuritiesCode'],
    how='inner'
)

if merged.empty:
    print("No matches—check your date ranges or mappings.")
    print("df1 dates:", df1['date'].unique())
    print("df2 dates:", df2['Date'].unique())
    print("Codes in df1:", df1['SecuritiesCode'].unique())
    print("Codes in df2:", df2['SecuritiesCode'].unique())
    exit()

print(f"Matched rows: {len(merged)}; companies: {merged['company name'].unique()}")

# 6. Correlate FinBERT sentiment
fin_corr, fin_p = pearsonr(merged['fin_score'], merged['Target'])
print(f"FinBERT score vs Target: corr={fin_corr:.4f}, p={fin_p:.4f}")

# 7. Correlate L–M lexicon ratios
for cat in ['positive','negative','uncertainty']:
    col = f'lm_{cat}_ratio'
    corr, p = pearsonr(merged[col], merged['Target'])
    print(f"L–M {cat}_ratio vs Target: corr={corr:.4f}, p={p:.4f}")

# 8. Embedding correlations (optional)
embed_cols = [c for c in merged.columns if c.startswith('finbert_emb_')]
embs = merged[embed_cols].values
corrs = [pearsonr(embs[:,i], merged['Target'])[0] for i in range(embs.shape[1])]
corrs = np.array(corrs)
print("Embedding dims →",
      f"mean={corrs.mean():.4f}, std={corrs.std():.4f},",
      f"min={corrs.min():.4f}, max={corrs.max():.4f}")

# 9. Mean-embedding vs Target
mean_emb = embs.mean(axis=1)
m_corr, m_p = pearsonr(mean_emb, merged['Target'])
print(f"Mean embedding vs Target: corr={m_corr:.4f}, p={m_p:.4f}")
