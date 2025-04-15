import requests
from bs4 import BeautifulSoup
import csv
import datetime

# Define the companies and their stock codes
companies = {
    "SMC Corporation": "6273.T",
    "Keyence Corporation": "6861.T",
    "Tokyo Electron Limited": "8035.T",
    "Fast Retailing Co., Ltd.": "9983.T",
    "Nintendo Co., Ltd.": "7974.T"
}

# Define the date range
start_date = datetime.date(2017, 1, 1)
end_date = datetime.date(2021, 12, 31)

# Define the output CSV file
output_file = 'news_data.csv'

# Function to fetch news data
def fetch_news(company_name, stock_code):
    news_data = []
    # Example URL for Google News search
    search_url = f"https://news.google.com/search?q={company_name}%20after:{start_date}%20before:{end_date}&hl=en-US&gl=US&ceid=US%3Aen"
    response = requests.get(search_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')
        for article in articles:
            headline = article.find('h3').text if article.find('h3') else 'No headline'
            link = article.find('a', href=True)['href'] if article.find('a', href=True) else 'No link'
            date_published = article.find('time')['datetime'] if article.find('time') else 'No date'
            # Fetch the full article content
            article_response = requests.get(link)
            if article_response.status_code == 200:
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                paragraphs = article_soup.find_all('p')
                content = ' '.join([para.text for para in paragraphs])
            else:
                content = 'Could not fetch article content'
            news_data.append({
                'date': date_published,
                'company': company_name,
                'headline': headline,
                'content': content
            })
    else:
        print(f"Failed to fetch news for {company_name}")
    return news_data

# Main script
all_news_data = []
for company, code in companies.items():
    print(f"Fetching news for {company}...")
    company_news = fetch_news(company, code)
    all_news_data.extend(company_news)

# Write to CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['date', 'company', 'headline', 'content'])
    writer.writeheader()
    for news in all_news_data:
        writer.writerow(news)

print(f"News data has been written to {output_file}")
