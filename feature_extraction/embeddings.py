import nltk
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon
nltk.download('vader_lexicon')
# Load the CSV file (replace 'output.csv' with your file path)
df = pd.read_csv('feature_extraction/financial_news_dec2021.csv')

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = df['contents'].tolist()
embeddings = model.encode(texts, show_progress_bar=True)
df['embeddings'] = list(embeddings)

# Calculate sentiment scores
sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['contents'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Save results
np.save('embeddings.npy', embeddings)
df[['date', 'company name', 'contents', 'sentiment_score']].to_csv('output_with_sentiment.csv', index=False)

# Preview
print(df[['date', 'company name', 'contents', 'sentiment_score']].head())
print(f"Embeddings saved with shape: {embeddings.shape}")