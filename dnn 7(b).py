import numpy as np
import matplotlib.pyplot as plt

# Generating simple sample data for demonstration
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to features
X_b = np.c_[np.ones((100, 1)), X]

# Initialize parameters (weights and bias)
theta = np.random.randn(2, 1)

# Hyperparameters
learning_rate = 0.01
iterations = 1000
stopping_threshold = 1e-5

# Gradient Descent
m = len(y)
for iteration in range(iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients
    # Calculate loss
    loss = np.mean((X_b.dot(theta) - y)**2)
    # Stopping criterion
    if loss < stopping_threshold:
        break

# Plotting the data and the fitted line
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression with Gradient Descent')
plt.grid(True)
plt.show()

print("Optimal parameters (weights and bias):", theta)
