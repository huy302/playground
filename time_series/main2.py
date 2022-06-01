import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from fbprophet import Prophet

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv', header=0)
df.plot()

# prepare expected column names
df.columns = ['ds', 'y']
df['ds']= pd.to_datetime(df['ds'])

# fit
model = Prophet()
model.fit(df)

# forecast
future = model.make_future_dataframe(periods=36, freq='M')
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# plot results
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# in-sample forecast
future = []
for i in range(1, 13):
	date = '1967-%02d' % i
	future.append([date])
future = pd.DataFrame(future)
future.columns = ['ds']
future['ds']= pd.to_datetime(future['ds'])
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# model eval
train_df = df.drop(df.index[-12:])
test_df = df.loc[df.index[-12:],:]
model = Prophet()
model.fit(train_df)
forecast = model.predict(test_df[['ds']])
y_true = test_df['y'].values
y_pred = forecast['yhat'].values
print(f'MAE = {mean_absolute_error(y_true, y_pred)}')
# plot expected vs actual
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()