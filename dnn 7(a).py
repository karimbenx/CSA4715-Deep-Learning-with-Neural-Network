import numpy as np
import matplotlib.pyplot as plt

# Generating sample data for demonstration
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Function to calculate the mean squared error (loss or cost)
def calculate_loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    loss = (1/(2*m)) * np.sum(np.square(predictions - y))
    return loss

# Function for gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations, stopping_threshold):
    m = len(y)
    losses = []
    for i in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradient
        loss = calculate_loss(X, y, theta)
        losses.append(loss)
        # Stopping criterion
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < stopping_threshold:
            break
    return theta, losses

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Initial parameters (weights and bias)
theta_initial = np.random.randn(2, 1)

# Hyperparameters
learning_rate = 0.01
iterations = 1000
stopping_threshold = 1e-5

# Gradient Descent
theta_final, losses = gradient_descent(X_b, y, theta_initial, learning_rate, iterations, stopping_threshold)

# Plotting the data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sample Data')
plt.grid(True)
plt.show()

# Plotting the loss curve
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid(True)
plt.show()

print("Optimal parameters (weights and bias):", theta_final)
