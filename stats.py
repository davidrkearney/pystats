import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X, y):
    """
    Performs linear regression on the data set X, y and plots the line of best fit
    """
    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Calculate the slope and y-intercept of the line of best fit
    m, b = np.polyfit(X.ravel(), y, 1)
    
    # Plot the data points and the line of best fit
    plt.scatter(X, y)
    plt.plot(X, m*X + b, color='red')
    plt.show()
    
    # Return the slope and y-intercept of the line of best fit
    return m, b

import numpy as np

# Generate random data for a simple linear regression problem
X = np.random.rand(100, 1)
y = 2*X + 1 + np.random.randn(100, 1)

# Call the linear_regression function with the random data
slope, intercept = linear_regression(X, y)


import numpy as np

# Generate random data for a simple linear regression problem
X = np.random.rand(100, 1)
y = 2*X + 1 + np.random.randn(100, 1)

# Call the linear_regression function with the random data
slope, intercept = linear_regression(X, y)

# Calculate the predicted values of y
y_pred = slope*X + intercept

# Calculate the mean squared error
mse = np.mean((y - y_pred)**2)

print("Mean Squared Error:", mse)

import numpy as np

# Generate random data for a simple linear regression problem
X = np.random.rand(100, 1)
y = 2*X + 1 + np.random.randn(100, 1)

# Call the linear_regression function with the random data
slope, intercept = linear_regression(X, y)

# Calculate the predicted values of y
y_pred = slope*X + intercept

# Calculate the total sum of squares
tss = np.sum((y - np.mean(y))**2)

# Calculate the residual sum of squares
rss = np.sum((y - y_pred)**2)

# Calculate the R-squared value
r_squared = 1 - (rss/tss)

print("R-squared value:", r_squared)

import numpy as np

# Generate random data for a simple linear regression problem
X = np.random.rand(100, 1)
y = 2*X + 1 + np.random.randn(100, 1)

# Call the linear_regression function with the random data
slope, intercept = linear_regression(X, y)

# Calculate the predicted values of y
y_pred = slope*X + intercept

# Calculate the total sum of squares
tss = np.sum((y - np.mean(y))**2)

# Calculate the residual sum of squares
rss = np.sum((y - y_pred)**2)

# Calculate the R-squared value
r_squared = 1 - (rss/tss)

print("R-squared value:", r_squared)