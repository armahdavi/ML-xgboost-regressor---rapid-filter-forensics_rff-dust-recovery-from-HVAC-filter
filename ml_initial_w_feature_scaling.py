# -*- coding: utf-8 -*-
"""
Program to run ML initial models for LR and XGB regressor and apply essential regularizations

@author: alima
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

exec(open(r'C:\PhD Research\Generic Codes\notion_corrections.py').read())
df = pd.read_excel(backslash_correct(r'C:\PhD Research\Paper 1 - Extraction\Processed\natural\ml_extraction_data.xlsx'))

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


#########################################################
### Step 2: ML Models Initial Run : Linear Regression ###
#########################################################

### Linear regression initial model
## Model creation and fitting to train set
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

## Making predictions
y_train_pred_lr = lin_reg.predict(X_train)
y_test_pred_lr = lin_reg.predict(X_test)

## Calculating and printing the scores for training and testing sets
train_score = r2_score(y_train, y_train_pred_lr)
test_score = r2_score(y_test, y_test_pred_lr)

print(f"Training set score: {train_score}")
print(f"Testing set score: {test_score}")



### Plotting the predicted and existing data: Linear Regression
## Graphing data in linear scale

x_line = np.linspace(0, 4, 100)  # Generating x values for the 1:1 line 
x_line_log = np.linspace(0.1, 10, 100)  # Generating x values for the 1:1 line (log scale)

plt.scatter(y_train, y_train_pred_lr, label = 'Train data', color = 'r')
plt.scatter(y_test, y_test_pred_lr, label = 'Test data', color = 'g')
plt.plot(x_line, x_line, color = 'b', linestyle = '--')

plt.xlabel('Measured Recoveries')
plt.xlim(0, 4)  # Setting x-axis limits

plt.ylabel('Predicted Recoveries')
plt.ylim(0, 4)  # Setting y-axis limits

plt.legend(edgecolor = 'black')

plt.savefig(r'C:\PhD Research\Paper 1 - Extraction\Processed\plots\rff\predicted_normal_lr.jpg', format = 'jpg', dpi = 800, bbox_inches = 'tight')
plt.show()


## Graphing data in logarithmic scale
plt.scatter(y_train, y_train_pred, label = 'Train data', color = 'r')
plt.scatter(y_test, y_test_pred, label = 'Test data', color = 'g')
plt.plot(x_line_log, x_line_log, color = 'b', linestyle = '--')

plt.xlabel('Measured Recoveries')
plt.xscale('log')
plt.xlim(0.1, 10)  

plt.ylabel('Predicted Recoveries')
plt.yscale('log')
plt.ylim(0.1, 10)  

plt.legend(edgecolor = 'black')

plt.savefig(r'C:\PhD Research\Paper 1 - Extraction\Processed\plots\rff\predicted_log_lr.jpg', format = 'jpg', dpi = 800, bbox_inches = 'tight')
plt.show()


#########################################################
### Step 3: ML Models Initial Run : XGBoost Regressor ###
#########################################################

### XGBoost regressor initial model
xgb_reg1 = xgb.XGBRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 3, random_state = 42)

## Fit the model to the training data
xgb_reg1.fit(X_train, y_train)

## Make predictions on the test data
y_train_pred_xg = xgb_reg1.predict(X_train)
y_test_pred_xg = xgb_reg1.predict(X_test)

## Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred_xg)
r2_train = r2_score(y_train, y_train_pred_xg)
mse_test = mean_squared_error(y_test, y_test_pred_xg)
r2_test = r2_score(y_test, y_test_pred_xg)
print('Mean squared error for train set is', mse_train)
print('R-squared for train set is', r2_train)

print('Mean squared error for test set is', mse_test)
print('R-squared for test set is', r2_test)



### XGBoost regressor initial model with mild regularization
xgb_reg2 = xgb.XGBRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 3, random_state = 42, 
                           colsample_bytree = 0.8)

## Note: The use of colsample_bytree = 0.8 to reduce overfitting slightly

## Fit the model to the training data
xgb_reg2.fit(X_train, y_train)

## Make predictions on the test data
y_train_pred_xg = xgb_reg2.predict(X_train)
y_test_pred_xg = xgb_reg2.predict(X_test)

## Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred_xg)
r2_train = r2_score(y_train, y_train_pred_xg)
mse_test = mean_squared_error(y_test, y_test_pred_xg)
r2_test = r2_score(y_test, y_test_pred_xg)
print('Mean squared error for train set is', mse_train)
print('R-squared for train set is', r2_train)

print('Mean squared error for test set is', mse_test)
print('R-squared for test set is', r2_test)

## Result: The use of colsample_bytree = 0.8 resulted in a slight reduction of overfitting by increasing the test set accuracy from 75% to 80%.


### Plotting the predicted and existing data: XGBoost Regressor
## Graphing data in linear scale
x_line = np.linspace(0, 4, 100)  # Generating x values for the 1:1 line 
x_line_log = np.linspace(0.1, 10, 100)  # Generating x values for the 1:1 line (log scale)

plt.scatter(y_train, y_train_pred_xg, label = 'Train data', color = 'r')
plt.scatter(y_test, y_test_pred_xg, label = 'Test data', color = 'g')
plt.plot(x_line, x_line, color = 'b', linestyle = '--')

plt.xlabel('Measured Recoveries')
plt.xlim(0, 4)  # Setting x-axis limits

plt.ylabel('Predicted Recoveries')
plt.ylim(0, 4)  # Setting y-axis limits

plt.legend(edgecolor = 'black')

plt.savefig(r'C:\PhD Research\Paper 1 - Extraction\Processed\plots\rff\predicted_normal_xg.jpg', format = 'jpg', dpi = 800, bbox_inches = 'tight')
plt.show()

## Graphing data in logarithmic scale
plt.scatter(y_train, y_train_pred_xg, label = 'Train data', color = 'r')
plt.scatter(y_test, y_test_pred_xg, label = 'Test data', color = 'g')
plt.plot(x_line_log, x_line_log, color = 'b', linestyle = '--')

plt.xlabel('Measured Recoveries')
plt.xscale('log')
plt.xlim(0.1, 10)  # Setting x-axis limits

plt.ylabel('Predicted Recoveries')
plt.yscale('log')
plt.ylim(0.1, 10)  # Setting y-axis limits

plt.legend()
plt.savefig(r'C:\PhD Research\Paper 1 - Extraction\Processed\plots\rff\predicted_log_xg.jpg', format = 'jpg', dpi = 800, bbox_inches = 'tight')
plt.show()


##############################################################
### Step 4: XGBoost Model Initial Run with Feature Scaling ###
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



