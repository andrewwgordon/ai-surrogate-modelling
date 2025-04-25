# Multi-fidelity Gaussian Process Regression using GPy

## Overview

This Python script demonstrates how to build and use a Multi-fidelity Gaussian Process (GP) model using the GPy library. Specifically, it implements a form of co-kriging where information from a computationally cheap, low-fidelity data source is combined with information from an expensive, high-fidelity data source to improve the predictive accuracy of the high-fidelity function.

## Purpose

The primary goal of multi-fidelity modeling is to leverage inexpensive low-fidelity data (e.g., from simulations with coarser meshes, simplified physics, or empirical models) to reduce the number of expensive high-fidelity data points (e.g., from high-resolution simulations or physical experiments) needed to build an accurate surrogate model of the high-fidelity process. This is particularly useful in engineering design, uncertainty quantification, and optimization where evaluating the high-fidelity function is time-consuming or costly.

## What this Script Demonstrates

* **Data Simulation:** Generating synthetic data for two related functions: a complex 'high-fidelity' function and a cheaper, noisier 'low-fidelity' approximation.
* **Modern GPy Implementation:** Using the current standard approach in GPy for multi-output/multi-fidelity GPs, which involves `GPy.models.GPRegression` combined with a product of a base kernel and a `GPy.kern.Coregionalize` kernel. This replaces older, potentially deprecated methods like `GPy.models.CoKriging`.
* **Data Preparation:** Structuring the input data (`X`) by adding an index column to identify the fidelity level (0 for high, 1 for low) and stacking the corresponding output data (`Y`).
* **Kernel Definition:** Constructing an appropriate kernel using `active_dims` to specify which input dimensions (original input vs. fidelity index) each part of the kernel acts upon.
* **Model Training:** Optimizing the hyperparameters of the combined kernel and the GP model.
* **Prediction:** Making predictions specifically for the high-fidelity output using the trained multi-fidelity model, including uncertainty estimates (variance).
* **Visualization:** Plotting the results, including the true underlying functions, the sparse high-fidelity training samples, the more abundant low-fidelity training samples, the model's predictions for the high-fidelity function, and the 95% confidence interval.

## How it Works

1.  **Simulation:** Two Python functions, `high_fidelity_function` and `low_fidelity_function`, are defined. The low-fidelity function is defined as a scaled and noisy version of the high-fidelity one. Sample data points are generated from both.
2.  **Data Splitting & Preparation:** A small number of high-fidelity samples and a larger number of low-fidelity samples are selected for training. The input arrays (`X_train_high`, `X_train_low`) are augmented with a second column indicating the fidelity index (0 or 1). These augmented arrays are stacked vertically into `X_train_augmented`. The corresponding output arrays (`y_train_high`, `y_train_low`) are also stacked vertically into `Y_train_stacked`.
3.  **Kernel Construction:**
    * A base kernel (`k_base`, e.g., Matern52) is defined to model the input space (acting on `active_dims=[0]`).
    * A `Coregionalize` kernel (`k_coreg`) is defined to model the correlation between the fidelity levels (acting on `active_dims=[1]`). It takes the number of outputs (`output_dim=2`) and a rank as parameters.
    * The final kernel is the element-wise product of `k_base` and `k_coreg`.
4.  **Model Initialization:** A `GPy.models.GPRegression` object is created using the `X_train_augmented`, `Y_train_stacked`, and the combined `kernel`.
5.  **Optimization:** The `model.optimize()` or `model.optimize_restarts()` method is called to find the optimal kernel hyperparameters (like lengthscale, variance, coregionalization parameters) by maximizing the log marginal likelihood.
6.  **Prediction:** To predict the high-fidelity output at new test points (`X_test`), these points are augmented with the high-fidelity index (0) -> `X_test_augmented`. The `model.predict()` method is called with this augmented input and `Y_metadata={'output_index': ...}` to specify that predictions for the output corresponding to index 0 are required.
7.  **Plotting:** Matplotlib is used to generate a comprehensive plot showing the true functions, the locations of the training samples (distinguished by fidelity), the model's prediction for the high-fidelity function, and the associated uncertainty.

## Requirements

* Python 3.x
* GPy
* NumPy
* Matplotlib

## Installation

You can install the required libraries using pip:

```bash
pip install GPy numpy matplotlib