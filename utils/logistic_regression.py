import numpy as np
from utils.training import train, mse

def logistic_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    return loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward(X, weights):
    return sigmoid(X @ weights.T)

def backward(X, y_true, y_pred):
    n_samples = X.shape[0]
    
    # Compute the gradient
    grad = (1 / n_samples) * (X.T @ (y_pred - y_true))
    
    return grad.flatten()

def train_sigmoid_regressor(X_train, y_train, epochs=30, lr=1e-1):
    return train(X_train, y_train, forward, backward, logistic_loss, epochs, lr)

def plot_2D_model(ax, X_train, y_train, X_test, y_test, epoch_parameters):
    y_pred = forward(X_train, epoch_parameters)

    # Plot the data points
    ax.scatter(X_train[:, 0], y_train, label='Train data', marker='P', alpha=0.3, s=20)
    ax.scatter(X_test[:, 0], y_test, label='Test data', color='red', alpha=0.3, marker='X', s=20)
    
    # Plot the learned function
    ax.plot(X_train[:, 0], y_pred, ':', linewidth=1.5, color='black', label='Learned $f(x)$', alpha=0.75)
    
    # Calculate the predicted probabilities for the range of x values
    y_probs = forward(X_train, epoch_parameters)
    
    # Find where the probability is approximately 0.5 (decision boundary)
    decision_boundary = X_train[np.abs(y_probs - 0.5).argmin()][0]
    
    # Plot the decision boundary as a vertical line
    ax.axvline(decision_boundary, color='blue', linestyle='--', label='Decision boundary', alpha=0.7)

