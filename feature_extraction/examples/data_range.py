import pandas as pd
import datetime

# Load the stock prices CSV
df = pd.read_csv('data/train_files/stock_prices.csv')  # Replace with your file path
df['date'] = pd.to_datetime(df['Date'])  # Adjust 'Date' to match your column name
print(df['date'])
# Define the date range
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime(2022, 1, 9)

# Filter data for the specified date range
mask = (df['date'] >= start_date) & (df['date'] <= end_date)

filtered_df = df[mask]
print(filtered_df)

# Generate all expected trading days (Monday to Friday)
all_trading_days = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' = business days
expected_days = len(all_trading_days)  # Should be ~22 days (Dec 1-30, weekdays only)
print(f"Expected trading days: {expected_days}")
print(f"Trading days: {all_trading_days}")
# Group by company (assuming 'SecuritiesCode' is the identifier; adjust if different)
company_counts = filtered_df.groupby('SecuritiesCode')['date'].nunique()
# Find companies with data for all expected trading days
complete_companies = company_counts[company_counts == expected_days].index.tolist()

# Map SecuritiesCode to company names (assuming you have a mapping CSV)
mapping_df = pd.read_csv('data/stock_list.csv')  # Your third CSV with SecuritiesCode and Name
mapping = dict(zip(mapping_df['SecuritiesCode'], mapping_df['Name']))
complete_company_names = [mapping.get(code, code) for code in complete_companies]

# Output results
print(f"Companies with complete data from 2021-12-01 to 2021-12-30 ({expected_days} days):")
for code, name in zip(complete_companies, complete_company_names):
    print(f"SecuritiesCode: {code}, Company Name: {name}")

# Additional check: Unique dates in filtered data
unique_dates = filtered_df['date'].dt.date.unique()
print(f"Unique dates in filtered data: {len(unique_dates)}")
print(f"Dates present: {sorted(unique_dates)}")