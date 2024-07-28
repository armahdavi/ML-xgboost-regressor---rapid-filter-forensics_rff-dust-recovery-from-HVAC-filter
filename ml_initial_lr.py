# -*- coding: utf-8 -*-
"""
Program to run initial linear regression ML model 

@author: alima
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

exec(open(r'C:\PhD Research\Generic Codes\notion_corrections.py').read())
df = pd.read_excel(backslash_correct(r'C:\PhD Research\Paper 1 - Extraction\Processed\natural\ml_extraction_data.xlsx'))

################################################
### ML Models Initial Run: Linear Regression ###
################################################

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
plt.title('Linear Regression ML', fontsize = 16)

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
plt.title('Linear Regression ML (Log)', fontsize = 16)

plt.savefig(r'C:\PhD Research\Paper 1 - Extraction\Processed\plots\rff\predicted_log_lr.jpg', format = 'jpg', dpi = 800, bbox_inches = 'tight')
plt.show()

