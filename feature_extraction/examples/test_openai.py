import requests
import pandas as pd
import re
from datetime import datetime, timedelta
import json

API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Load company name from stock_list.csv
df = pd.read_csv('data/stock_list.csv')
companies = {row['Name']: row['SecuritiesCode'] for _, row in df.iterrows()}
# Only keep the first 20 companies for testing
companies = {k: v for i, (k, v) in enumerate(companies.items()) if i < 20}

# companies = {"Toyota":1301, "Sony":1302, "Nintendo":1303, "Panasonic":1304, "Mitsubishi":1305}

companies = {"Toyota":1301}

start = "2020-12-01"
end = "2020-12-30"
results = []  # Will store dictionaries for each news entry
raw_data = {}

# Helper function to extract news in the correct format
def extract_news(company_name, response_text):
    news_entries = []
    # Regex to capture:
    # - Index (e.g., "1.")
    # - Company name
    # - Date (YYYY-MM-DD)
    # - Contents (everything until the URL)
    # - URL
    pattern = re.compile(
        r"(\d+\.)\s*([^,]+),\s*(\d{4}-\d{2}-\d{2}),\s*([^,]+),\s*(https?://[^\s\n]+)(?:\n|$)",
        re.MULTILINE
    )
    
    matches = pattern.findall(response_text)
    for match in matches:
        _, company, date, contents, url = match
        news_entries.append({
            'date': date.strip(),
            'company': company_name.strip(),
            'contents': contents.strip(),
            'url': url.strip()
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

        query = f"Provide concise financial news contents with news published date stamps (YYYY-MM-DD) and brief context for {company_name} from {current.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}.\
             And your output should be a list, and each element should be in the format of 'company name', 'YYYY-MM-DD', 'contents', 'url'"

        messages = [
            {"role": "system", "content": "You're a financial news analyst providing contents, news, their URLs, and their exact date stamps."},
            {"role": "user", "content": query}
        ]

        payload = {
            "model": "gpt-4o",  # Using gpt-4o-mini for cost efficiency
            "messages": messages,
            "max_tokens": 2000
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        raw_data[company_name] = data['choices'][0]['message']['content']
        if 'choices' in data and len(data['choices']) > 0:
            answer_text = data['choices'][0]['message']['content']
            extracted_news = extract_news(company_name, answer_text)
            results.extend(extracted_news)
            print(f"Extracted news for {company_name} from {current.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}")
            print(extracted_news)
        
        current = month_end + timedelta(days=1)

# Create a DataFrame from results and save it to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("financial_news_2020.csv", index=False)

print("Successfully saved to financial_news_2020.csv")
# Dump raw data to a file
with open('raw_data.json', 'w') as f:
    json.dump(raw_data, f)
print("Successfully saved raw data to raw_data.json")