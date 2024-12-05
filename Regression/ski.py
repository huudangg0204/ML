from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

x= np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y= np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T


one = np.ones((x.shape[0], 1))
xbar = np.concatenate((one, x), axis = 1)
print(xbar)
print(y)
# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
