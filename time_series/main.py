import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from fbprophet import Prophet

# get data
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
fig = plt.figure(figsize=(18,4))
plt.gca().xaxis.set_major_locator(mdates.DayLocator((1,100)))
plt.setp(plt.gca().get_xticklabels(), rotation=60, ha="right")
plt.plot(df['ds'], df['y'])
plt.show()

# fit
m = Prophet()
m.fit(df)

# forecast
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# plot results
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)