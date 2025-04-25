# -*- coding: utf-8 -*-
import GPy
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate high-fidelity and low-fidelity data
def high_fidelity_function(X):
    # Ensure X is treated as a 1D array for the formula
    X_flat = X.flatten()
    # Original formula logic
    result = np.sin(X_flat) * (1 - (X_flat - np.pi)**2) * np.exp(-(X_flat - np.pi)**2)
    # Reshape back to original X shape, ensuring it's always 2D (N, 1)
    return result.reshape(-1, 1)

def low_fidelity_function(X):
    # Calls high_fidelity_function which now returns (N, 1)
    hf_output = high_fidelity_function(X)
    # Perform calculation ensuring shapes are compatible
    result = (1/3) * hf_output + 0.1 * np.random.randn(*X.shape)
    return result # Shape will be (N, 1)

# Generate sample data
X = np.linspace(0, 2 * np.pi, 100)[:, None] # Shape (100, 1)
y_high_fidelity = high_fidelity_function(X) + 0.01 * np.random.randn(*X.shape) # Shape (100, 1)
y_low_fidelity = low_fidelity_function(X) # Shape (100, 1)

# Split data into training and testing sets
# Use fewer high-fidelity points to make the multi-fidelity aspect more useful
high_fidelity_indices = np.random.choice(np.arange(100), size=10, replace=False)
low_fidelity_indices = np.random.choice(np.arange(100), size=50, replace=False)
test_indices = np.setdiff1d(np.arange(100), np.union1d(high_fidelity_indices, low_fidelity_indices)) # Ensure test points are not in training

# High-fidelity training data
X_train_high = X[high_fidelity_indices]
y_train_high = y_high_fidelity[high_fidelity_indices]

# Low-fidelity training data
X_train_low = X[low_fidelity_indices]
y_train_low = y_low_fidelity[low_fidelity_indices]

# Test data (using the true high-fidelity function for comparison)
X_test = X[test_indices]
y_test_true = high_fidelity_function(X_test) # True function for comparison

# --- Prepare data for Coregionalized Model ---
# Augment X data with an index column to indicate fidelity level
# 0 for high-fidelity, 1 for low-fidelity
X_train_augmented = np.vstack((np.hstack((X_train_high, np.zeros_like(X_train_high))),
                               np.hstack((X_train_low, np.ones_like(X_train_low)))))

# Stack Y data vertically
Y_train_stacked = np.vstack((y_train_high, y_train_low))

# --- Create the Coregionalized Gaussian Process model ---

# Define the base kernel for the input space (X)
# input_dim=1 because our original X is 1-dimensional
# active_dims=[0] because this kernel operates on the first column (the original X)
k_base = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=1., active_dims=[0])

# Define the coregionalization kernel
# input_dim=1 because it operates on the index column
# output_dim=2 because we have two fidelity levels (high and low)
# rank=1 assumes a simple linear relationship between fidelities (common for co-kriging)
# active_dims=[1] because this kernel operates on the second column (the fidelity index)
k_coreg = GPy.kern.Coregionalize(input_dim=1, output_dim=2, rank=1, active_dims=[1])

# Combine the kernels using element-wise multiplication
kernel = k_base * k_coreg

# Create the GPRegression model with the augmented data and combined kernel
# Note: We no longer use GPy.models.CoKriging
model = GPy.models.GPRegression(X_train_augmented, Y_train_stacked, kernel)

# --- Optional: Fix or constrain parameters ---
# Sometimes fixing the coregionalization parameters can help stability
# model['Coregionalize.W'].fix() # Example: Fix W matrix
# model['Coregionalize.kappa'].fix() # Example: Fix kappa vector

# --- Optimize the model ---
print("Optimizing model...")
model.optimize(messages=True)
# For potentially better results, try restarting the optimization
# model.optimize_restarts(num_restarts=5, messages=True, robust=True)
print("Optimization finished.")
print(model)


# --- Plotting (model.plot() is less informative for coregionalized models) ---
# The default model.plot() plots both outputs; we'll create a custom plot
# model.plot()
# plt.suptitle("GPy Default Coregionalized Model Plot")
# plt.show()

# --- Make predictions for the HIGH-FIDELITY output ---
# Create the prediction points matrix, augmenting X_test with the high-fidelity index (0)
X_test_augmented = np.hstack((X_test, np.zeros_like(X_test)))

# Define the metadata required for prediction with coregionalized models
# This tells the model which output index (fidelity level) to predict for each input point
output_index_predict = X_test_augmented[:, 1:] # Shape (n_test, 1), contains zeros

# Predict using the high-fidelity index
y_pred_high, y_var_high = model.predict(X_test_augmented, Y_metadata={'output_index': output_index_predict.astype(int)})

# --- Custom Plot Results ---
plt.figure(figsize=(12, 7))

# Plot true functions
plt.plot(X.flatten(), high_fidelity_function(X).flatten(), 'r-', label='True High-fidelity f(x)', linewidth=2)
plt.plot(X.flatten(), low_fidelity_function(X).flatten(), 'b--', label='True Low-fidelity f(x)', linewidth=1.5, alpha=0.7)

# Plot training samples
plt.scatter(X_train_high.flatten(), y_train_high.flatten(), c='r', s=80, marker='o', label='High-fidelity samples', zorder=5)
plt.scatter(X_train_low.flatten(), y_train_low.flatten(), c='b', s=40, marker='s', label='Low-fidelity samples', zorder=4, alpha=0.8)

# Plot predictions for high-fidelity
sort_indices = np.argsort(X_test.flatten()) # Sort for plotting lines correctly
plt.plot(X_test.flatten()[sort_indices], y_pred_high.flatten()[sort_indices], 'k-', label='High-fidelity Prediction', linewidth=2)
plt.fill_between(X_test.flatten()[sort_indices],
                 (y_pred_high - 1.96 * np.sqrt(y_var_high)).flatten()[sort_indices],
                 (y_pred_high + 1.96 * np.sqrt(y_var_high)).flatten()[sort_indices],
                 color='gray', alpha=0.3, label='95% Confidence Interval')

# Plot true test points for comparison
plt.scatter(X_test.flatten(), y_test_true.flatten(), c='lime', s=50, marker='x', label='True High-fidelity (Test Points)', zorder=6)


plt.xlabel("Input X")
plt.ylabel("Output Y")
plt.title('Multi-fidelity GP using Coregionalized Kernel')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('plot.png')