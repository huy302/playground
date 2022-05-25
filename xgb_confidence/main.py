import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

plt.style.use('ggplot')

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
alpha = 0.9 # 90% confidence interval

# Quantile prediction with GradientBoostingRegressor
clf = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=500)
clf.fit(X_train, y_train)
y_upper = clf.predict(X_test)

clf.set_params(alpha=1.0 - alpha)
clf.fit(X_train, y_train)
y_lower = clf.predict(X_test)

clf.set_params(loss='ls')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({'P10': y_upper, 'P50': y_pred, 'real': y_test, 'P90': y_lower}).sort_values(by=['P50']).to_csv('gbr_result.csv', index=False)

confidence = sum([r >= l and r <= u for l, r, u in zip(y_lower, y_test, y_upper)]) / len(y_test) * 100
print(f'real confidence = {confidence} vs reference confidence = {alpha}')

# Quantile prediction with XGBRegressor

# log cosh quantile is a regularized quantile loss function
def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile

clf = XGBRegressor(objective=log_cosh_quantile(alpha),n_estimators=100)
clf.fit(X_train, y_train)
y_upper = clf.predict(X_test)

clf = XGBRegressor(objective=log_cosh_quantile(1.0-alpha),n_estimators=100)
clf.fit(X_train, y_train)
y_lower = clf.predict(X_test)

clf = XGBRegressor(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({'P10': y_upper, 'P50': y_pred, 'real': y_test, 'P90': y_lower}).sort_values(by=['real']).to_csv('gbr_result.csv', index=False)

confidence = sum([r >= l and r <= u for l, r, u in zip(y_lower, y_test, y_upper)]) / len(y_test) * 100
print(f'real confidence = {confidence} vs reference confidence = {alpha}')