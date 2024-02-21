import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:/Users/Merwin S/OneDrive/Documents/WPS Cloud Files/395703879/data(1).csv")

# Choose independent and dependent variables
X = data['radius_mean'].values.reshape(-1, 1)  # Independent variable
y = data['area_mean']  # Dependent variable

# Perform linear regression
regression_model = LinearRegression()
regression_model.fit(X, y)

# Get regression coefficients
b0 = regression_model.intercept_
b1 = regression_model.coef_[0]

# Print coefficients
print("Intercept (b0):", b0)
print("Slope (b1):", b1)

# Predict y values
y_pred = regression_model.predict(X)

# Plot the regression line
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(X, y_pred, color='blue', label='Regression Line')
plt.xlabel('Radius Mean')
plt.ylabel('Area Mean')
plt.title('Linear Regression')
plt.legend()
plt.show()
