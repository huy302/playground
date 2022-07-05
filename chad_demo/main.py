# Generating a demo dataset for Chad
# data size: 8,760 hours x 30,000 assets, x 30 years x 4+ forecasts

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

n_assets = 10

hour_list = []
asset_list = []
year_list = []
forecast_list = []

# one_year_hour_list = list(range(1, 365*24 + 1))
# asset_hour_list = []
# for fc in range(1, 5):
#     for i in range(30):
#         asset_hour_list += one_year_hour_list
hour_list = list(range(1, 365*24*30 + 1))
asset_hour_list = []
for fc in range(1, 5):
    asset_hour_list += hour_list
asset_year_list = []
for fc in range(1, 5):
    for i in range(30):
        asset_year_list += list(np.full(365*24, i+1))

def generate_asset(asset_id):
    n_val_per_forecast = 365*24*30
    values = None
    forecast_values = None
    asset_id_values = np.full(n_val_per_forecast * 4, f'asset {asset_id}')
    for fc in range(1, 5):
        values_fc = list(np.cumsum(np.random.rand(n_val_per_forecast) - 0.5) + np.random.random_sample() * 100)
        forecast_values_fc = list(np.full(n_val_per_forecast, f'forecast {fc}'))
        if values is None:
            values = values_fc
        else:
            values = np.append(values, values_fc)
        if forecast_values is None:
            forecast_values = forecast_values_fc
        else:
            forecast_values = np.append(forecast_values, forecast_values_fc)
    return asset_id_values, forecast_values, values


results = Parallel(n_jobs=-1)(delayed(generate_asset)(asset_id) for asset_id in tqdm(range(n_assets)))
len(results[0][0])

all_hour = []
all_year = []
all_asset = []
all_forecast = []
all_values = []
for i in tqdm(range(n_assets)):
    all_hour += asset_hour_list
    all_year += asset_year_list
    all_asset += list(results[i][0])
    all_forecast += list(results[i][1])
    all_values += list(results[i][2])

df = pd.DataFrame({'Hour': all_hour, 'Year': all_year, 'Asset': all_asset, 'Forecast': all_forecast, 'Values': all_values})
df.to_csv('power.csv', index=False)