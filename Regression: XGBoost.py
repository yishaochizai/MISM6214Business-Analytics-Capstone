# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 15:00:17 2022

@author: kchen
"""
## the packages we are using
import pandas as pd
import numpy as np
import xgboost as xgb
from math import sqrt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from yellowbrick.regressor import prediction_error
from xgboost import plot_importance
## we are running the same code in all 3 XGBoost models
## the only difference is reading different classification csv files

# here is an example for decision tress with XGBoost regressor
# prediction_dt is our previous decision tree classification results

df = pd.read_csv('prediction_dt.csv')
prediction_result = df[df.y_pred != 0]
prediction_result = prediction_result.drop(columns = ['y_pred','Index'])

# Setting X,Y variables 
X = prediction_result.drop(['d30_spend'], axis=1)
y = prediction_result.d30_spend

# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)

# Train and test set are converted to DMatrix objects,
# as it is required by learning API.
train_dmatrix = xgb.DMatrix(data = X_train, label = y_train)
test_dmatrix = xgb.DMatrix(data = X_test, label = y_test)
# Parameter dictionary specifying base learner
param = {"booster":"gbtree", "objective":"reg:linear"}

xgb_r = xgb.train(params = param, dtrain = train_dmatrix, num_boost_round = 10)
pred = xgb_r.predict(test_dmatrix)

# RMSE Computation

rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE : % f" %(rmse))

# R2 Computation
print("Coefficient of determination: %.2f" % r2_score(y_test, pred))

# Feature Importance
xgb.plot_importance(xgb_r, max_num_features=10)



