# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:06:29 2024

@author: alima
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from matplotlib import pyplot as plt
exec(open('C:/Life/5- Career & Business Development/Learning/Python Practice/Generic Codes/notion_corrections.py').read())


################################################
### XGBoost with HP tuning and 5-fold CV (2) ###
################################################

## Reading dataset and keep the upper 95% dust load data
df = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - Extraction\Processed\natural\ml_extraction_data.xlsx'))
threshold = df['dustmass'].quantile(0.1)

df = df[df['dustmass'] > threshold]
X, y = df[['Site_N', 'Round_N', 'ft', 'Cycle_N', 'runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max']], df['M_t']

## Create the XGBoost regressor model
xgb_reg = xgb.XGBRegressor()

## Define the hyperparameter grid for grid search
param_grid = {
    'n_estimators': [10, 50, 100, 200, 300],
    'learning_rate': [1, 0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7, 9, 11],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

## Perform grid search with cross-validation
grid_search = GridSearchCV(estimator = xgb_reg, param_grid = param_grid, cv = 5, scoring = 'r2')
grid_search.fit(X, y)

## Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

cv_scores = cross_val_score(best_model, X, y, cv = 5, scoring = 'r2')

## Evaluate the best model on the test data
y_pred = best_model.predict(X)
r2_mean = np.mean(cv_scores)
r2_std = np.std(cv_scores)

## Print the best hyperparameters and model performance
print("Best Hyperparameters All:", best_params)
print("R-squared mean:", r2_mean)
print("R-squared standard deviation:", r2_std)


## Plot y_pred vs. y_test
plt.scatter(y, y_pred)
plt.xlabel('Measured Values')
plt.ylabel('Predicted Values')
plt.xscale('log')
plt.yscale('log')
plt.title('True Values vs. Predicted Values')

## Add 1:1 line
max_value = max(max(y), max(y_pred))
min_value = min(min(y), min(y_pred))
plt.plot([min_value, max_value], [min_value, max_value], color = 'red', linestyle = '--')

plt.show()

