import GPy
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate high-fidelity and low-fidelity data
def high_fidelity_function(X):
    return np.sin(X) * (1 - (X - np.pi)**2) * np.exp(-(X - np.pi)**2)

def low_fidelity_function(X):
    return (1/3) * high_fidelity_function(X) + 0.1 * np.random.randn(*X.shape)

# Generate sample data
X = np.linspace(0, 2 * np.pi, 100)[:, None]
y_high_fidelity = high_fidelity_function(X) + 0.01 * np.random.randn(*X.shape)
y_low_fidelity = low_fidelity_function(X)

# Split data into training and testing sets
train_indices = np.arange(50)
test_indices = np.arange(50, 100)

X_train_high, X_train_low = X[train_indices], X[train_indices]
y_train_high, y_train_low = y_high_fidelity[train_indices], y_low_fidelity[train_indices]

X_test = X[test_indices]
y_test = y_high_fidelity[test_indices]

# Create a Gaussian Process model using GPy
kernel = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=1.)

# Create a coKriging model (linear GP model)
model = GPy.models.CoKriging(X_train_high, y_train_high, X_train_low, y_train_low, kernel=kernel)

# Optimize the model
model.optimize(messages=True)

# Plot results
model.plot()
plt.show()

# Make predictions
y_pred, y_var = model.predict(X_test)
plt.figure(figsize=(10, 5))
plt.plot(X, y_high_fidelity, 'r--', label='High-fidelity data')
plt.plot(X, y_low_fidelity, 'b--', label='Low-fidelity data')
plt.scatter(X_train_high, y_train_high, c='r', s=50, label='High-fidelity samples')
plt.scatter(X_train_low, y_train_low, c='b', s=50, label='Low-fidelity samples')
plt.plot(X_test, y_pred, 'g-', label='Predictions')
plt.fill_between(X_test.flatten(), 
                 (y_pred - 1.96 * np.sqrt(y_var)).flatten(), 
                 (y_pred + 1.96 * np.sqrt(y_var)).flatten(), 
                 color='g', alpha=0.2, label='Confidence interval')
plt.legend()
plt.title('Multi-fidelity GP CoKriging')
plt.show()
