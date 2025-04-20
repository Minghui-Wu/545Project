import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# Define headers with fake user-agent for anti-crawling
headers = {'User-Agent': UserAgent().random}

# List of sites to crawl
target_sites = [
    'site:nintendo.com',
    'site:nintendolife.com',
    'site:polygon.com',
    'site:famitsu.com',
    'site:ign.com'
]

# Function to fetch news with anti-anti crawling measures
def fetch_nintendo_news(query, year, month, language='en'):
    results = []
    for site in target_sites:
        search_url = f'https://www.google.com/search?q={query}+{site}&hl={language}&tbs=cdr:1,cd_min:{month}/1/{year},cd_max:{month}/31/{year}'
        response = requests.get(search_url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch page from {site}: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')

        print(f"soup: {soup}")

        for g in soup.select('div.tF2Cxc'):
            title = g.select_one('h3').text if g.select_one('h3') else 'No title'
            link = g.select_one('a')['href'] if g.select_one('a') else 'No link'
            snippet = g.select_one('div.IsZvec').text if g.select_one('div.IsZvec') else 'No snippet'
            results.append({'title': title, 'link': link, 'snippet': snippet})

    return results

# Example usage
news_results = fetch_nintendo_news('Nintendo', 2021, 12, language='en')

for idx, result in enumerate(news_results, 1):
    print(f"News {idx}: {result['title']}\nLink: {result['link']}\nSnippet: {result['snippet']}\n")