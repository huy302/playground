from turtle import title
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
        print(y_pred)
        print(y_true)
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile

# TODO: try https://github.com/scikit-learn/scikit-learn/blob/89c835234e1f161bff5ebd3bb6bc5fa100c917e5/sklearn/ensemble/_gb_losses.py#L173

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


alpha = 0.9 # 90% confidence interval
# XGBoost with gb loss as objective function
# log cosh quantile is a regularized quantile loss function
def gb_loss_quantile(alpha):
    def _gb_loss_quantile(y_true, y_pred):
        # y_pred = y_pred.ravel()
        # diff = y_true - y_pred
        # mask = y_true >= y_pred
        # # loss = (alpha * diff[mask].sum() - (1 - alpha) * diff[~mask].sum()) / len(y_true)
        # grad = [(1 - alpha) if v else alpha for v in mask]
        # hess = np.full(len(y_true), 1.0)
        # return grad, hess
        # # # return loss
        x = y_true - y_pred
        grad = (x<(alpha-1.0))*(1.0-alpha)-((x>=(alpha-1.0))& (x<alpha) )*x-alpha*(x>alpha)
        hess = ((x>=(alpha-1.0))& (x<alpha) ) 
        return grad,hess

    return _gb_loss_quantile

from typing import Tuple
def softprob_obj(labels: np.ndarray, predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows = labels.shape[0]
    grad = np.zeros((rows, classes), dtype=float)
    hess = np.zeros((rows, classes), dtype=float)
    eps = 1e-6
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            g = p[c] - 1.0 if c == target else p[c]
            h = max((2.0 * p[c] * (1.0 - p[c])).item(), eps)
            grad[r, c] = g
            hess[r, c] = h

    grad = grad.reshape((rows * classes, 1))
    hess = hess.reshape((rows * classes, 1))
    return grad, hess

clf = XGBRegressor(objective=gb_loss_quantile(alpha),n_estimators=100)
clf.fit(X_train, y_train)
y_upper = clf.predict(X_test)

clf = XGBRegressor(objective=gb_loss_quantile(1.0-alpha),n_estimators=100)
clf.fit(X_train, y_train)
y_lower = clf.predict(X_test)

clf = XGBRegressor(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({'P10': y_upper, 'P50': y_pred, 'real': y_test, 'P90': y_lower}).sort_values(by=['real']).to_csv('gbr_result.csv', index=False)

confidence = sum([r >= l and r <= u for l, r, u in zip(y_lower, y_test, y_upper)]) / len(y_test) * 100
print(f'real confidence = {confidence} vs reference confidence = {alpha}')