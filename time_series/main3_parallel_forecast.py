import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from fbprophet import Prophet
from joblib import Parallel, delayed
from tqdm import tqdm
tqdm.pandas()

# EDA
df = pd.read_csv('data\\train.csv', header=0)
df.head()
df.isnull().sum()
df.nunique()

# single model fit
model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
)

history_df = df[['date', 'sales']]
history_df.columns = ['ds', 'y']
history_df['ds'] = pd.to_datetime(history_df['ds'])
model.fit(history_df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
)
forecast_pd = model.predict(future_pd)
predict_fig = model.plot(forecast_pd, xlabel='date', ylabel='sales')

# multiple models, one for each store/item
def groupby_apply_parallel(df: pd.DataFrame,
    groupby_col: list,
    func: callable,
    *args,
    **kwargs) -> pd.DataFrame:
    '''Perform group by and apply on dataframe in parallel fashion

    Args:
        df (pd.DataFrame): Input dataframe
        groupby_col (str): Column name to perform GroupBy on
        func (function): Function to be executed on each group
        *args: Additional values to be passed into :func:`func`
        **kwargs: Additional keywords to be passed into :func:`func`

    Returns:
        pd.DataFrame: Result dataframe
    '''
    group_obj = df.groupby(by=groupby_col, sort=False)
    group_list = Parallel(n_jobs=-1)(
        delayed(func)(group, *args, **kwargs) for _, group in tqdm(group_obj))
    if len(group_list) > 0:
        return pd.concat(group_list, sort=False).reset_index(drop=True)
    return df

def forecast_store_item(history_df: pd.DataFrame) -> pd.DataFrame:
    history_df = history_df.dropna()
    # instantiate the model, configure the parameters
    model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    # fit the model
    model.fit(history_df)
    # forecast
    future = model.make_future_dataframe(periods=90, freq='d')
    forecast_df = model.predict(future)
    # assemble results and return
    f_df = forecast_df[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].set_index('ds')
    h_df = history_df.set_index('ds')

    # return h_df.join(f_df, how='left').reset_index() # doing this will drop forecasted rows
    return f_df.join(h_df, how='left').reset_index()

store_item_df = df.copy(deep=True).rename(columns={'date': 'ds', 'sales': 'y'})
store_item_df['ds'] = pd.to_datetime(store_item_df['ds'])
results = groupby_apply_parallel(store_item_df, ['store', 'item'], forecast_store_item)

# plot 1 store 1 item
store1_item1 = results[(results.store == 1) & (results.item == 1)]
store1_item1[['y', 'yhat', 'yhat_upper', 'yhat_lower']].plot()

# eval result
y_df = results[['y', 'yhat']].dropna()
y_true = y_df['y'].values
y_pred = y_df['yhat'].values
print(f'Accuracy = {(1-sum(abs(y_true-y_pred))/sum(y_true))*100} (%)')