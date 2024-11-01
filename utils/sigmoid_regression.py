import numpy as np
from utils.training import train, mse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward(X, weights):
    return sigmoid(X @ weights.T)

def backward(X, y_true, y_pred):
    error = y_pred - y_true
    sigmoid_derivative = y_pred * (1 - y_pred)
    
    # Chain rule to compute the gradient w.r.t weights
    grad = error * sigmoid_derivative
    
    # Gradient w.r.t weights
    [grad_weights] = grad.T @ X
    
    return grad_weights

def train_sigmoid_regressor(X_train, y_train, epochs=30, lr=1e-1):
    return train(X_train, y_train, forward, backward, mse, epochs, lr)

def plot_2D_model(ax, X_train, y_train, X_test, y_test, epoch_parameters):
    y_pred = forward(X_train, epoch_parameters)

    ax.scatter(X_train[:, 0], y_train, label='Train data', marker='P', alpha=0.3, s=20)
    ax.scatter(X_test[:, 0], y_test, label='Test data', color='red', alpha=0.3, marker='X', s=20)
    ax.plot(X_train[:, 0], y_pred, ':', linewidth=1.5, color='black', label='Learned $f(x)$', alpha=0.75)