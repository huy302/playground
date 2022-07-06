# Generating a demo dataset for Chad
# data size: 8,760 hours x 30,000 assets, x 30 years x 4+ forecasts

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from configure_logging import configure_logging
from snowflake_db_manager import Snowflake_DBManager

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

if __name__ == "__main__":
    configure_logging()

    n_assets = 1000

    hour_list = []
    asset_list = []
    year_list = []
    forecast_list = []

    db_manager = Snowflake_DBManager('HUYNA_TEST')

    hour_list = list(range(1, 365*24*30 + 1))
    asset_hour_list = []
    for fc in range(1, 5):
        asset_hour_list += hour_list
    asset_year_list = []
    for fc in range(1, 5):
        for i in range(30):
            asset_year_list += list(np.full(365*24, i+1))

    results = Parallel(n_jobs=-1)(delayed(generate_asset)(asset_id) for asset_id in tqdm(range(n_assets)))
    len(results[0][0])

    # write each asset to snowflake
    for i in range(n_assets):
        logging.info(f'============ Processing asset {i} ============')
        asset_df = pd.DataFrame({
            'Hour': asset_hour_list,
            'Year': asset_year_list,
            'Asset': list(results[i][0]),
            'Forecast': list(results[i][1]),
            'Vals': list(results[i][2])
        })
        db_manager.write_db(asset_df, 'POWER_ASSETS', f'asset {i}')