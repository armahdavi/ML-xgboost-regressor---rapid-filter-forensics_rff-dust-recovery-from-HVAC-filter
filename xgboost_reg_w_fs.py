# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:59:40 2024

@author: alima
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


exec(open(r'C:\PhD Research\Generic Codes\notion_corrections.py').read())
df = pd.read_excel(backslash_correct(r'C:\PhD Research\Paper 1 - Extraction\Processed\natural\ml_extraction_data.xlsx'))

### Overal features and targets
X, y = df[['Site_N', 'Round_N', 'ft', 'Cycle_N', 'runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max']], df['M_t']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


###############################
### Step 1: Feature Scaling ###
###############################

### Overal features and targets
X, y = df[['Site_N', 'Round_N', 'ft', 'Cycle_N', 'runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max']], df['M_t']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

### Feature scaling based on MinMax
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


##############################################################
### Step 2: XGBoost Model Initial Run with Feature Scaling ###
##############################################################

xgb_reg3 = xgb.XGBRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 3, random_state = 42, 
                           colsample_bytree = 0.8)

## Note: The use of colsample_bytree = 0.8 to reduce overfitting slightly

## Fit the model to the training data
xgb_reg3.fit(X_train_scaled, y_train)

## Make predictions on the test data
y_train_pred_xg = xgb_reg3.predict(X_train_scaled)
y_test_pred_xg = xgb_reg3.predict(X_test_scaled)

## Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred_xg)
r2_train = r2_score(y_train, y_train_pred_xg)
mse_test = mean_squared_error(y_test, y_test_pred_xg)
r2_test = r2_score(y_test, y_test_pred_xg)
print('Mean squared error after feature scaling for train set is', mse_train)
print('R-squared after feature scaling for train set is', r2_train)

print('Mean squared error after feature scaling for test set is', mse_test)
print('R-squared after feature scaling for test set is', r2_test)

## Result: Feature scaling does not change the scores. The relatively small dataset causes a fast convergence regardless of feature scaling.
