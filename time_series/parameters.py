"""
@author: Jiajun Jiang
@file: parameters.py
@time: 9/21/2022 9:56 AM
@desc:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from functools import partial
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import scipy
plt.style.use('ggplot')

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
alpha = 0.9 # 90% confidence interval

class XGBQuantile(XGBRegressor):
    def __init__(self, quant_alpha=0.95, quant_delta=1.0, quant_thres=1.0, quant_var=1.0, base_score=0.5,
                 booster='gbtree', colsample_bylevel=1, colsample_bynode=1,
                 colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1,
                 missing=1, n_estimators=100, enable_categorical=False, gpu_id=0,
                 n_jobs=1, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, importance_type='weight',
                 scale_pos_weight=1,  subsample=1, interaction_constraints=None, monotone_constraints=None,
                 num_parallel_tree=1, predictor = 'cpu_predictor', tree_method='auto', validate_parameters=False,
                 verbosity=0):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var

        super().__init__(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
                         colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate,
                         max_delta_step=max_delta_step, colsample_bynode = colsample_bynode,
                         max_depth=max_depth, min_child_weight=min_child_weight, missing=missing,
                         n_estimators=n_estimators, enable_categorical=enable_categorical, gpu_id=gpu_id,
                         n_jobs=n_jobs,  objective=objective, random_state=random_state, importance_type=importance_type,
                         reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
                         subsample=subsample, interaction_constraints=interaction_constraints,
                         monotone_constraints=monotone_constraints, num_parallel_tree=num_parallel_tree,
                         predictor = predictor, tree_method=tree_method, validate_parameters=validate_parameters,
                         verbosity=verbosity)

        self.test = None

    def fit(self, X, y):
        super().set_params(objective=partial(XGBQuantile.quantile_loss, alpha=self.quant_alpha, delta=self.quant_delta,
                                             threshold=self.quant_thres, var=self.quant_var))
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        y_pred = super().predict(X)
        score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
        score = 1. / score
        return score

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                    (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
                    2 * np.random.randint(2, size=len(y_true)) - 1.0) * var
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess

    @staticmethod
    def original_quantile_loss(y_true, y_pred, alpha, delta):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                    (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
        return grad, hess

    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantile.quantile_cost(x=y_true - y_pred, alpha=alpha)
        score = np.sum(score)
        return score

    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)

    @staticmethod
    def get_split_gain(gradient, hessian, l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i]) ** 2 / (np.sum(hessian[:i]) + l) + np.sum(gradient[i:]) ** 2 / (
                        np.sum(hessian[i:]) + l) - np.sum(gradient) ** 2 / (np.sum(hessian) + l))

        return np.array(split_gain)


estimator = XGBQuantile(quant_alpha = 1-alpha,quant_delta= 5.0, quant_thres = 5.0 ,quant_var=4.0)

gs  = RandomizedSearchCV(estimator = estimator,
                        param_distributions={
                                  'quant_delta':scipy.stats.uniform(0.01,10.0),
                                  'quant_var':scipy.stats.uniform(1.0,200.0),
                                  'quant_thres':scipy.stats.uniform(0.01,400.0)
                        },n_iter=100,cv=5,return_train_score =False, verbose=1)

gs.fit(X_train,y_train)