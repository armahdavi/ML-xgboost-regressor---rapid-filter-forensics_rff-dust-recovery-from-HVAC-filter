# -*- coding: utf-8 -*-
"""
Program to run XGBoost with hyperparameter tuning 

@author: alima
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
exec(open('C:/Life/5- Career & Business Development/Learning/Python Practice/Generic Codes/notion_corrections.py').read())


exec(open('C:/Life/5- Career & Business Development/Learning/Python Practice/Generic Codes/notion_corrections.py').read())
df = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - Extraction\Processed\natural\ml_extraction_data.xlsx'))


#########################################################
### Pre-train modelling based on previous development ###
#########################################################
### Overal features and targets
X, y = df[['Site_N', 'Round_N', 'ft', 'Cycle_N', 'runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max']], df['M_t']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

xgb_reg3 = xgb.XGBRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 3, random_state = 42, 
                           colsample_bytree = 0.8)
## Note: The use of colsample_bytree = 0.8 to reduce overfitting slightly

## Fit the model to the training data
xgb_reg3.fit(X_train, y_train)


############################################
### XGBoost with HP tuning and 5-fold CV ###
############################################

## Reading dataset
df = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - Extraction\Processed\natural\ml_extraction_data.xlsx'))
X, y = df[['Site_N', 'Round_N', 'ft', 'Cycle_N', 'runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max']], df['M_t']
X_train_test, X_dev, y_train_test, y_dev = train_test_split(X, y, test_size = 0.2, random_state = 42)

## Create the XGBoost regressor model and hyperparameter grid
xgb_reg4 = xgb.XGBRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7, 9, 11],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

## Perform grid search with cross-validation over cross-validated (train/test) data
grid_search = GridSearchCV(estimator = xgb_reg4, param_grid = param_grid, cv = 5, scoring = 'r2')
grid_search.fit(X_train_test, y_train_test)

## Get the best hyperparameters and model 
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

## Evaluate the best model on the cross-validated (train/test) and dev sets
cv_scores_xgb4 = cross_val_score(best_model, X_train_test, y_train_test, cv = 5, scoring = 'r2') # getting the 5-fold scores
y_pred_train_test_xgb4 = best_model.predict(X_train_test) 
r2_mean_xgb4 = np.mean(cv_scores_xgb4) # getting the 5-fold scores mean
r2_std_xgb4 = np.std(cv_scores_xgb4) # getting the 5-fold scores stdev

y_pred_dev_xgb4 = best_model.predict(X_dev) # prediction over dev set
r2_dev_xgb4 = r2_score(y_dev, y_pred_dev_xgb4) # score of dev set

## Print the best hyperparameters and model performance for training set
print("Best Hyperparameters:", best_params)
print("R-squared mean of best model:", round(r2_mean_xgb4, 2))
print("R-squared STDEV of best model:", round(r2_std_xgb4, 2))
print("R-squared of dev set:", round(r2_dev_xgb4, 2))


## Evaluate the xgb3 model (previously tested) on the cross-validated (train/test) and dev sets
cv_scores_xgb3 = cross_val_score(xgb_reg3, X_train_test, y_train_test, cv = 5, scoring = 'r2') # getting the 5-fold scores
# y_pred_train_test_xgb3 = xgb_reg3.predict(X_train_test)
r2_mean_xgb3 = np.mean(cv_scores_xgb3) # getting the 5-fold scores mean
r2_std_xgb3 = np.std(cv_scores_xgb3) # getting the 5-fold scores stdev

y_pred_dev_xgb3 = best_model.predict(X_dev) # prediction over dev set
r2_dev_xgb3 = r2_score(y_dev, y_pred_dev_xgb3) # score of dev set

print("R-squared mean of the pre-trained model:", round(r2_mean_xgb3, 2))
print("R-squared STD of the pre-trained model:", round(r2_std_xgb3, 2))
print("R-squared dev set of the pre-trained model:", round(r2_dev_xgb3, 2))

### Plotting the predicted values over measured
y_pred_all = best_model.predict(X)
plt.scatter(y_train_test, y_pred_train_test_xgb4, color = 'r', label = 'Cross-validated Set')
plt.scatter(y_dev, y_pred_dev_xgb4, color = 'g', label = 'Dev Set')
plt.xlabel('Measured Recoveries')
plt.xlim(0.01, 10)  # Setting x-axis limits
plt.xscale('log')

plt.ylabel('Predicted Recoveries')
plt.ylim(0.01, 10)  # Setting y-axis limits
plt.yscale('log')

plt.legend(edgecolor = 'black')

## Add 1:1 line
x_line_log = np.linspace(0.01, 10, 100)  # Generating x values for the 1:1 line (log scale)
plt.plot(x_line_log, x_line_log, color = 'b', linestyle = '--')

plt.show()
