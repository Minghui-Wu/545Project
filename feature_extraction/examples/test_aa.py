from meteostat import Point, Daily
from datetime import datetime

# Define location (Ann Arbor)
ann_arbor = Point(42.2808, -83.7430)

# Define date
start = datetime(2024, 7, 1)
end = datetime(2024, 7, 30)

# Fetch daily data
data = Daily(ann_arbor, start, end)
data = data.fetch()

print(data)  # Outputs temperature, precipitation, etc.