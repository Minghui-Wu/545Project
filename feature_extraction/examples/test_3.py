import re

# Raw data
raw_data = """
Unfortunately, the search results do not provide specific financial news headlines for SMC Corporation (6273.T) between 2021-12-01 and 2021-12-30. However, I can provide some general information about SMC Corporation and suggest where you might find relevant news.

**General Information:**
- SMC Corporation is a leading manufacturer of automatic control equipment and other industrial devices.
- It is listed on the Tokyo Stock Exchange under the ticker 6273.T.

**Potential Sources for News:**
- For specific financial news headlines, you might want to check financial news websites such as Bloomberg, Reuters, or Nikkei Asia.

Since there are no specific headlines available from the search results, here is a placeholder for what the format would look like if news were available:

[YYYY-MM-DD, SMC Corporation, Headline, Brief Context, URL]

If you need specific news, I recommend checking the aforementioned financial news platforms for archives from December 2021.
Here are concise financial news headlines for Keyence Corporation (6861.T) from 2021-12-01 to 2021-12-30:

1. **2021-12-02**: Keyence Corporation, New 90-day high: JP¥53,980  
   - Keyence reached a new 90-day high in its stock price, reflecting strong market performance.  
   [URL: https://simplywall.st/stocks/jp/tech/tse-6861/keyence-shares](https://simplywall.st/stocks/jp/tech/tse-6861/keyence-shares)

2. **2021-12-17**: Keyence Corporation, New 90-day high: JP¥55,220  
   - Keyence continued its upward trend, achieving another new 90-day high in stock price.  
   [URL: https://simplywall.st/stocks/jp/tech/tse-6861/keyence-shares](https://simplywall.st/stocks/jp/tech/tse-6861/keyence-shares)

3. **2021-12-20**: Keyence Corporation, Financial Update  
   - For the nine months ended December 20, 2021, Keyence reported a 45% increase in revenues to ¥545.33 billion and a 65% increase in net income to ¥218.41 billion.  
   [URL: https://capital.com/en-ae/markets/shares/keyence-corporation-share-price](https://capital.com/en-ae/markets/shares/keyence-corporation-share-price)

4. **2021-12-29**: Keyence Corporation, Q3 Results Announcement Scheduled  
   - Keyence is set to report its Q3 results on January 29, 2022, following the end of the period on December 31, 2021.  
   [URL: https://simplywall.st/stocks/jp/tech/tse-6861/keyence-shares](https://simplywall.st/stocks/jp/tech/tse-6861/keyence-shares)

Note: The specific news for December 29, 2021, regarding the Q3 results announcement, is more about the upcoming event rather than a financial news headline from December itself. However, it is relevant to the period as it sets the stage for future financial reporting.
Unfortunately, the search results do not provide specific financial news headlines for Tokyo Electron Limited (8035.T) between 2021-12-01 and 2021-12-30. However, I can provide some general information about the company during that period based on available data:

1. **General Performance**: Tokyo Electron Limited continued to be a significant player in the semiconductor and flat panel display equipment market. The company's financial performance for the fiscal year ending March 31, 2021, was previously reported, with expectations for net sales of ¥1,399,102 million and net income of ¥242,941 million[2].

2. **Market Trends**: The semiconductor industry experienced fluctuations due to supply chain issues and demand for advanced technology. However, specific news for Tokyo Electron during December 2021 is not detailed in the search results.

Given the lack of specific news headlines within the search results, here is a placeholder for what such headlines might look like if they were available:

- **[YYYY-MM-DD, Tokyo Electron Limited, Headline, Brief Context, URL]**

For precise financial news headlines, it would be best to consult financial news platforms or Tokyo Electron's official investor relations website for press releases and announcements during that period.
Unfortunately, the search results do not provide specific financial news headlines for Fast Retailing Co., Ltd. (9983.T) between 2021-12-01 and 2021-12-30. However, I can provide some general information about the company during that period based on available data:

1. **General Performance**: Fast Retailing, the parent company of UNIQLO, typically releases financial reports on a quarterly basis. The fiscal year for Fast Retailing ends on August 31, so during December 2021, the company would have been in the middle of its fiscal year.

2. **No Specific News**: Without specific news articles from the search results, it's challenging to provide exact headlines. However, Fast Retailing's financial performance and business updates are usually reported in their quarterly earnings releases and annual reports.

If you need more detailed financial news from that period, you might want to check financial news websites or Fast Retailing's official investor relations page for press releases and financial reports from that time. 

Here is a general format of what the output might look like if specific news were available:

- [YYYY-MM-DD, Fast Retailing Co., Ltd., Headline, Brief Context, URL]

Given the lack of specific news in the search results, this format cannot be filled with exact data for the requested period.
Here are concise financial news headlines for Nintendo Co., Ltd. (7974.T) from 2021-12-01 to 2021-12-30:

1. **2021-12-01, Nintendo Co., Ltd.**  
   **Headline:** Nintendo Continues Strong Sales Despite Pandemic Slowdown  
   **News:** Nintendo's sales remain robust, driven by the success of the Nintendo Switch and popular titles like Animal Crossing: New Horizons and Pokémon games.  
   **URL:** Not available for this specific date, but general information can be found on Nintendo's financial reports.

2. **2021-12-18, Nintendo Co., Ltd.**  
   **Headline:** The NPD Group Reports U.S. Video Game Industry Sales  
   **News:** The NPD Group reported that the Nintendo Switch was the best-selling hardware platform in units sold for December 2021. Pokémon Brilliant Diamond/Shining Pearl was the top-selling game on Nintendo platforms for the month.  
   **URL:** https://www.npd.com/wps/portal/npd/us/news/press-releases/2022/npd-group-reports-u-s-video-game-industry-sales-for-december-2021/

3. **2021-12-31, Nintendo Co., Ltd.**  
   **Headline:** Nintendo Switch Achieves Milestone Sales  
   **News:** As of December 2021, the Nintendo Switch has sold over 103.5 million units worldwide, solidifying its position as Nintendo's best-selling console.  
   **URL:** Not available for this specific date, but general information can be found in Nintendo's financial reports.

For more detailed financial reports, you can refer to Nintendo's official financial statements or news articles from reputable sources like GamesIndustry.biz and Statista.
"""

# Regular expression pattern to extract news items
news_pattern = re.compile(
    r"\*\*(\d{4}-\d{2}-\d{2})\*\*: ([^,]+), ([^\n]+)\n\s*-\s*([^[]+)\n\s*\[URL: ([^\]]+)\]"
)

# Extract matched data
news_entries = news_pattern.findall(raw_data)

print(news_entries)

# Formatting output
formatted_news = []
for entry in news_entries:
    date, company, headline, context, url = entry
    formatted_news.append(f"{date} | {company} | {headline} - {context} ({url})")

# Display output
for news in formatted_news:
    print(news)
