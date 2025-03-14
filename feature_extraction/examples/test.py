import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import re

def fetch_google_news(date_range, security_codes, output_file="news.txt"):
    start_date, end_date = date_range
    base_url = "https://www.google.com/search?q={query}&tbs=cdr:1,cd_min:{start},cd_max:{end}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    perplexity_api_key = "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"
    
    code_to_ticker = {
        "1312": ("SMC Corporation", "6273.T"),
        "1313": ("KEYENCE CORPORATION", "6861.T"),
        "1319": ("Tokyo Electron Limited", "8035.T"),
        "1320": ("FAST RETAILING CO.,LTD.", "9983.T"),
        "1321": ("Nintendo Co.,Ltd.", "7974.T")
    }
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_str = start_dt.strftime("%m/%d/%Y")
    end_str = end_dt.strftime("%m/%d/%Y")
    
    all_news = []
    
    for sec_code in security_codes:
        if sec_code not in code_to_ticker:
            all_news.append(f"{start_date} - {sec_code} - Error: Unknown security code")
            continue
            
        company_name, ticker = code_to_ticker[sec_code]
        query = f"{company_name} financial news {ticker}"
        url = base_url.format(query=query, start=start_str, end=end_str)
        
        try:
            # Step 1: Scrape Google News
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            news_items = soup.find_all("div", class_="SoaBEf") or soup.find_all("div", class_="Gx5Zad")
            if not news_items:
                news_items = soup.select("div a[href*='/news/']")
            
            print(f"{sec_code}: Found {len(news_items)} items from {url}")
            
            if news_items:
                for item in news_items:
                    title_tag = item.find("div", role="heading") or item.find("h3") or item.find("a")
                    title = title_tag.text.strip() if title_tag else "No Title"
                    
                    date_tag = item.find("span", class_="r0bn4c") or item.find("span")
                    reported_date = date_tag.text.strip() if date_tag else "Unknown Date"
                    try:
                        parsed_reported_date = datetime.strptime(reported_date, "%b %d, %Y").strftime("%Y-%m-%d")
                    except ValueError:
                        parsed_reported_date = reported_date
                    
                    # Step 2: Use Perplexity to verify company and infer event date
                    messages = [
                        {"role": "system", "content": "You are a financial news analyst. Verify if this news pertains to the specified company and extract the exact event date (not reported date) if mentioned."},
                        {"role": "user", "content": f"News item for {company_name}: '{title}'. Reported date: {reported_date}. Does this pertain to {company_name}? If yes, what is the event date (YYYY-MM-DD)?"}
                    ]
                    pplx_response = requests.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers={"Authorization": f"Bearer {perplexity_api_key}", "Content-Type": "application/json"},
                        json={"model": "sonar", "messages": messages, "max_tokens": 100, "temperature": 0.7}
                    )
                    pplx_data = pplx_response.json()
                    response_text = pplx_data.get("choices", [{}])[0].get("message", {}).get("content", "Unable to verify or extract event date")
                    
                    # Check if news pertains to the company
                    if "yes" in response_text.lower() or "likely" in response_text.lower():
                        # Extract event date from Perplexity response
                        date_match = re.search(r"\d{4}-\d{2}-\d{2}", response_text)
                        if date_match:
                            event_date = date_match.group(0)
                            event_dt = datetime.strptime(event_date, "%Y-%m-%d")
                            if start_dt <= event_dt <= end_dt:
                                all_news.append(f"{event_date} - {sec_code} - {title}")
                            else:
                                continue  # Skip if event date is outside range
                        else:
                            # Fallback to reported date if no event date found, with verification
                            try:
                                event_dt = datetime.strptime(parsed_reported_date, "%Y-%m-%d")
                                if start_dt <= event_dt <= end_dt:
                                    all_news.append(f"{parsed_reported_date} - {sec_code} - {title} (Reported date used - event date unclear)")
                            except ValueError:
                                all_news.append(f"{parsed_reported_date} (Unparsed) - {sec_code} - {title} (Reported date used - event date unclear)")
            
            if not any(sec_code in n for n in all_news):
                all_news.append(f"{start_date} - {sec_code} - No financial events reported for {company_name} during this period")
        
        except requests.exceptions.RequestException as e:
            all_news.append(f"{start_date} - {sec_code} - Error: Failed to fetch news - {str(e)}")
        
        time.sleep(2)
    
    # Sort by date
    all_news.sort(key=lambda x: datetime.strptime(x.split(" - ")[0], "%Y-%m-%d") if "Unparsed" not in x else start_dt)
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"=== Financial News Report ({start_date} to {end_date}) ===\n\n")
        for news in all_news:
            file.write(f"{news}\n\n")

# Example usage
security_codes = ["1312", "1313", "1319", "1320", "1321"]
date_range = ("2021-12-06", "2022-12-30")
fetch_google_news(date_range, security_codes, "news.txt")