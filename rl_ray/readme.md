Update:

1. Modify __init__() and _get_observation() to shrink observations into one dimension
2. Revise reset() to make it compatible with ray
3. Revise step() to make it compatible with ray. Here I set truncate always False and only adjust terminate. You can modify it accordingly. 