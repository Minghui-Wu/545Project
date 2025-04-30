from eventregistry import *
import csv

API_KEY = "9c424819-88de-4756-a16d-a81e0779ba49"  # 替换为你自己的 API Key
er = EventRegistry(apiKey=API_KEY)

# 设置要查询的日本公司名
companies = ["Toyota", "Sony", "Nintendo", "Panasonic", "Mitsubishi"]

# 时间范围
# start_date = "2021-12-01"
# end_date = "2021-12-31"

start_date = "2025-03-21"
end_date = "2025-04-15"

# 获取日本的 URI
jp_uri = er.getLocationUri("Japan")

# 准备输出文件
with open("japan_companies_news_dec_2021.csv", mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "company", "content", "url"])

    for company in companies:
        print(f"Fetching news for: {company}")
        q = QueryArticlesIter(
            keywords=QueryItems.AND([company]),
            sourceLocationUri=jp_uri,
            dateStart=start_date,
            dateEnd=end_date,
            dataType=["news"],
        )

        # 限制最大数量，可自行调整
        for article in q.execQuery(er, sortBy="date", maxItems=300):
            date = article.get("date", "")[:10]
            content = article.get("body", "")
            url = article.get("url", "")
            writer.writerow([date, company, content, url])
