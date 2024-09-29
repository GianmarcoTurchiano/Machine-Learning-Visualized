import numpy as np
from utils.training import train, mse

def forward(X, weights):
    y = X @ weights.T
    return y

def backward(X, y, y_pred):
    errors = y - y_pred
    return -2 * (errors @ X) / len(X)

def train_linear_regressor(X_train, y_train, epochs=30, lr=1e-1):
    forward2 = lambda X, weights: forward(X, weights).flatten()
    return train(X_train, y_train, forward2, backward, mse, epochs, lr)

def train_linear_regressor_multivariate(X_train, y_train, epochs=30, lr=1e-1):
    backward2 = lambda X, y, y_pred: backward(X, y.T, y_pred.T)
    return train(X_train, y_train, forward, backward2, mse, epochs, lr)

def plot_2D_model(ax, X_train, y_train, X_test, y_test, epoch_parameters):
    y_pred = forward(X_train, epoch_parameters)

    ax.scatter(X_train[:, 0], y_train, label='Train data', marker='P', alpha=0.3, s=20)
    ax.scatter(X_test[:, 0], y_test, label='Test data', color='red', alpha=0.3, marker='X', s=20)
    ax.plot(X_train[:, 0], y_pred, ':', linewidth=1.5, color='black', label='Learned $f(x)$', alpha=0.75)

def plot_3D_plane_model(ax, X_train, y_train, X_test, y_test, epoch_parameters):
    x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
    x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max()

    x1_range = np.linspace(x1_min, x1_max, 100)
    x2_range = np.linspace(x2_min, x2_max, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Flatten the grid for prediction
    x_grid_flat = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

    # Get predicted y values for the meshgrid using the current epoch's parameters
    y_pred_grid = forward(x_grid_flat, epoch_parameters).reshape(x1_grid.shape)

    # Plot the data
    ax.plot_wireframe(x1_grid, x2_grid, y_pred_grid, color='black', alpha=0.4, rstride=20, cstride=20, linestyle='--', zorder=1)
    ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color='black', alpha=0.1, rstride=100, cstride=100, edgecolor='none', label=r'Learned $f(x_1, x_2)$', zorder=1)

    # Plot the training and test data as scatter plots
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, label="Train data", marker='P', alpha=0.3, s=20, zorder=2)
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="Test data", color='red', marker='X', alpha=0.3, s=20, zorder=2)

def plot_3D_line_model(ax, X_train, y_train, X_test, y_test, epoch_parameters):
    epoch_parameters = epoch_parameters.T
    y_pred = forward(X_train, epoch_parameters)

    ax.scatter(y_train[:, 0], y_train[:, 1], X_train[:, 0], label='Train data', marker='P', alpha=0.3, s=20)
    ax.scatter(y_test[:, 0], y_test[:, 1], X_test[:, 0], label='Test data', color='red', alpha=0.3, marker='X', s=20)

    ax.plot(y_pred[:, 0], y_pred[:, 1], X_train[:, 0], ':', linewidth=1.5, color='black', label='Learned $f(x)$', alpha=0.75, zorder=20)
