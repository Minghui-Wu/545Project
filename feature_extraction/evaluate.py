# calculate the correlations between the features extracted from the texts and the labels
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load the first CSV with embeddings and sentiment scores
# Assuming it’s saved as 'output_with_sentiment.csv' and embeddings as 'embeddings.npy'
df1 = pd.read_csv('output_with_sentiment.csv')
embeddings = np.load('embeddings_2020.npy', allow_pickle=True)
# embeddings = np.load('embeddings_2020.npy', allow_pickle=True)

df1['embeddings'] = list(embeddings)

# Load the second CSV with Target values
# df2 = pd.read_csv('data/supplemental_files/stock_prices.csv')  # Replace with your file name
df2 = pd.read_csv('data/train_files/stock_prices.csv')
# 3. Third CSV with SecuritiesCode to company name mapping
df3 = pd.read_csv('data/stock_list.csv')  # Replace with your file name

# Create a mapping from SecuritiesCode to company name
mapping = dict(zip(df3['SecuritiesCode'], df3['Name']))

# Add SecuritiesCode to df1 based on mapping (reverse mapping from company name to code)
reverse_mapping = {v: k for k, v in mapping.items()}
df1['SecuritiesCode'] = df1['company name'].map(reverse_mapping)

print(f"securities code: {df1['SecuritiesCode'].unique()}")

# Standardize date formats
df1['date'] = pd.to_datetime(df1['date']).dt.strftime('%Y-%m-%d')
df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%Y-%m-%d')

print(df1)

# Merge df1 (embeddings/sentiment) with df2 (Target) on 'date' and 'SecuritiesCode'
merged_df = pd.merge(
    df1[['date', 'SecuritiesCode', 'company name', 'contents', 'sentiment_score', 'embeddings']],
    df2[['Date', 'SecuritiesCode', 'Target']],
    left_on=['date', 'SecuritiesCode'],
    right_on=['Date', 'SecuritiesCode'],
    how='inner'
)

# Check merged data
print("Merged DataFrame:")
print(merged_df[['date', 'SecuritiesCode', 'company name', 'sentiment_score', 'Target']])
print(f"Number of matched rows: {len(merged_df)}")

# print matched companies
matched_companies = merged_df['company name'].unique()

print(f"Matched companies: {matched_companies}")

# If no matches, provide feedback
if len(merged_df) == 0:
    print("No matches found. Checking date and SecuritiesCode overlaps:")
    print("Dates in df1:", df1['date'].unique())
    print("Dates in df2:", df2['Date'].unique())
    print("SecuritiesCodes in df1:", df1['SecuritiesCode'].dropna().unique())
    print("SecuritiesCodes in df2:", df2['SecuritiesCode'].unique())
else:
    # Extract data for correlation
    sentiment_scores = merged_df['sentiment_score'].values
    targets = merged_df['Target'].values
    embeddings = np.vstack(merged_df['embeddings'].values)  # Convert list of embeddings to 2D array

    # Correlation between sentiment scores and Target
    sentiment_corr, sentiment_pval = pearsonr(sentiment_scores, targets)
    print(f"Sentiment Score vs Target Correlation: {sentiment_corr:.4f}, p-value: {sentiment_pval:.4f}")

    # Correlation between each embedding dimension and Target
    embedding_corrs = []
    for i in range(embeddings.shape[1]):  # Loop over each dimension
        corr, pval = pearsonr(embeddings[:, i], targets)
        embedding_corrs.append(corr)

    # Summary of embedding correlations
    embedding_corrs = np.array(embedding_corrs)
    print(f"Embedding Correlations (mean across dimensions): {np.mean(embedding_corrs):.4f}")
    print(f"Embedding Correlations (std across dimensions): {np.std(embedding_corrs):.4f}")
    print(f"Embedding Correlations (min): {np.min(embedding_corrs):.4f}, (max): {np.max(embedding_corrs):.4f}")

    # Reduce embeddings to mean and correlate
    embedding_means = np.mean(embeddings, axis=1)
    mean_embedding_corr, mean_embedding_pval = pearsonr(embedding_means, targets)
    print(f"Mean Embedding vs Target Correlation: {mean_embedding_corr:.4f}, p-value: {mean_embedding_pval:.4f}")