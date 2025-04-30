import json
import pandas as pd

# Load your data
with open('news.txt', 'r', encoding='utf-8') as file:
    raw_content = file.read()

# Extract the JSON data from the file content
data_start = raw_content.find('{')
data_json = raw_content[data_start:]
data = json.loads(data_json)

articles = data['articles']

# Extract required fields from the articles
processed_articles = []
for article in articles:
    processed_articles.append({
        'title': article.get('title'),
        'url': article.get('url'),
        'publishedAt': article.get('pubDate'),
        'summary': article.get('summary') or article.get('description'),
        'source': article.get('source', {}).get('domain')
    })

# Create a DataFrame
df = pd.DataFrame(processed_articles)

# Save results to CSV for further analysis
df.to_csv('processed_nintendo_news_dec2021.csv', index=False)

# Display sample articles
print(df.head())
