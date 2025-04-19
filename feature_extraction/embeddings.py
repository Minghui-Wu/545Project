import nltk
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon
nltk.download('vader_lexicon')
# Load the CSV file (replace 'output.csv' with your file path)
# df = pd.read_csv('/nvme1/Projects/545Project/data/TOYOTA_2020_News.csv')
df = pd.read_csv('/nvme1/Projects/545Project/data/news_data_2020_standardized.csv')
# df = pd.read_csv('/nvme1/Projects/545Project/data/SONY_2020_News.csv')
# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = df['contents'].tolist()
embeddings = model.encode(texts, show_progress_bar=True)
df['embeddings'] = list(embeddings)

# Calculate sentiment scores
sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['contents'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Save results
np.save('embeddings_2020.npy', embeddings)
df[['date', 'company name', 'contents', 'sentiment_score']].to_csv('output_with_sentiment.csv', index=False)

# Preview
print(df[['date', 'company name', 'contents', 'sentiment_score']].head())
print(f"Embeddings saved with shape: {embeddings.shape}")