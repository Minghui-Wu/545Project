import requests
import pandas as pd
import re
from datetime import datetime, timedelta

API_URL = "https://api.perplexity.ai/chat/completions"
API_KEY = "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"  

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Load company name from stock_list.csv
df = pd.read_csv('data/stock_list.csv')
companies = {row['Name']: row['SecuritiesCode'] for _, row in df.iterrows()}
# only keep the first 2 companies for testing
companies = {k: v for i, (k, v) in enumerate(companies.items()) if i < 20}

start = "2021-12-01"
end = "2021-12-30"
results = []  # will store dictionaries for each news entry

raw_data = {}
# Helper function to extract news in the correct format
def extract_news(company_name, response_text):
    news_entries = []
    # Split the response text into blocks that start with a date
    blocks = re.findall(r"(\d{4}-\d{2}-\d{2}:[\s\S]+?)(?=\n\d{4}-\d{2}-\d{2}:|\Z)", response_text)
    
    # Regex to capture:
    # - Date
    # - Headline (optionally followed by a citation like [1])
    # - Context (starting with a hyphen or 'Context:')
    # - Optional URL line in the format "[URL: ...]"
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}):\s*(.*?)\s*(?:\[\d+\])?\s*\n(?:-|\*\*Context:\*\*|Context:)\s*(.*?)(?:\n\[URL:\s*([^\]]+)\])?(?:\n|$)",
        re.DOTALL
    )
    
    for block in blocks:
        match = pattern.search(block)
        if match:
            date, headline, context, url = match.groups()
            url = url.strip() if url and url.strip() else "No URL available"
            news_entries.append({
                'date': date,
                'company': company_name,
                'contents': f"{headline.strip()} - {context.strip()}",
                'url': url
            })
    return news_entries

# Iterate through each company and fetch news
for company_name, ticker in companies.items():
    current = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    while current <= end_date:
        month_end = (current + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        if month_end > end_date:
            month_end = end_date

        query = f"Provide concise financial news headlines with date stamps (YYYY-MM-DD) and brief context for {company_name} from {current.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}."

        messages = [
            {"role": "system", "content": "You're a financial news analyst providing headlines, news, their URLs and their exact date stamps."},
            {"role": "user", "content": query}
        ]

        payload = {
            "model": "sonar-pro",
            "messages": messages,
            "max_tokens": 500
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        print(f"data: {data}")
        raw_data[company_name] = data
        if 'choices' in data and len(data['choices']) > 0:
            answer_text = data['choices'][0]['message']['content']
            extracted_news = extract_news(company_name, answer_text)
            results.extend(extracted_news)
            print(f"Extracted news for {company_name} from {current.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}")
            print(extracted_news)
        
        current = month_end + timedelta(days=1)

# Create a DataFrame from results and save it to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("financial_news_2021.csv", index=False)

print("Successfully saved to financial_news_2021.csv")
# dump raw data to a file
import json
with open('raw_data.json', 'w') as f:
    json.dump(raw_data, f)
print("Successfully saved raw data to raw_data.json")
