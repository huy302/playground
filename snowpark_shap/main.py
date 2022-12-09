import pandas as pd
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
import shap
import parameters as p

# train a dummy model
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
clf = XGBRegressor()
clf.fit(X, y)

# deploy snowpark model
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import *
from snowflake.snowpark.functions import udf
from snowflake.snowpark import version
print(f'Snowpark version {version.VERSION}')

snowflake_conn_prop = p.snowflake_conn_prop
session = Session.builder.configs(snowflake_conn_prop).create()
print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
print(session.sql('create stage if not exists MODELSTAGE').collect())

features = list(diabetes['feature_names'])
session.add_packages('numpy', 'pandas', 'scikit-learn==1.1.1', 'xgboost==1.5.0', 'shap==0.39.0')
@udf(name='diabetes_predict', is_permanent=True, stage_location='@MODELSTAGE', replace=True, session=session)
def diabetes_predict(args: list) -> list: # this one does not work
    row = pd.DataFrame([args], columns=features)
    pred = clf.predict(row)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(row)
    return [pred] + list(shap_values.reshape(-1))
# def diabetes_predict(args: list) -> float: # this one works
#     row = pd.DataFrame([args], columns=features)
#     pred = clf.predict(row)
#     return pred