from utils.sigmoid_regression import sigmoid, sigmoid_derivative
from utils.training import train_neural, mse
import numpy as np

def forward(X, weights1, weights2, activation_fn):
    hidden = activation_fn(X @ weights1.T)  # Hidden layer
    y_pred = hidden @ weights2.T            # Output layer
    return hidden, y_pred

def backward(X, hidden, y, y_pred, weights2, activation_der):
    output_error = y_pred - y
    hidden_error = output_error @ weights2 * activation_der(hidden)
    
    grad_w2 = output_error.T @ hidden / len(y)
    grad_w1 = hidden_error.T @ X / len(y)
    
    return grad_w1, grad_w2

def train_neural_regressor_sigmoid(X_train, y_train, epochs=30, lr=1e-1):
    forward_sigmoid = lambda X, weights1, weights2: forward(X, weights1, weights2, sigmoid)
    backward_sigmoid = lambda X, hidden, y, y_pred, weights2: backward(X, hidden, y, y_pred, weights2, sigmoid_derivative)

    return train_neural(X_train, y_train, forward_sigmoid, backward_sigmoid, mse, epochs, lr)

def plot_2D_sigmoid_model(ax, X_train, y_train, X_test, y_test, epoch_parameters):
    plot_2D_model(ax, X_train, y_train, X_test, y_test, epoch_parameters, sigmoid)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def train_neural_regressor_relu(X_train, y_train, epochs=30, lr=1e-1):
    forward_relu = lambda X, weights1, weights2: forward(X, weights1, weights2, relu)
    backward_relu = lambda X, hidden, y, y_pred, weights2: backward(X, hidden, y, y_pred, weights2, relu_derivative)

    return train_neural(X_train, y_train, forward_relu, backward_relu, mse, epochs, lr)

def plot_2D_relu_model(ax, X_train, y_train, X_test, y_test, epoch_parameters):
    plot_2D_model(ax, X_train, y_train, X_test, y_test, epoch_parameters, relu)

def plot_2D_model(ax, X_train, y_train, X_test, y_test, epoch_parameters, activation_fn):
    w_1 = epoch_parameters[0][0]
    w_2 = epoch_parameters[0][1]

    _, y_pred = forward(X_train, np.array([[w_1]]), np.array([[w_2]]), activation_fn)
    
    ax.scatter(X_train[:, 0], y_train, label='Train data', marker='P', alpha=0.3, s=20)
    ax.scatter(X_test[:, 0], y_test, label='Test data', color='red', alpha=0.3, marker='X', s=20)
    ax.plot(X_train[:, 0], y_pred, ':', linewidth=1.5, color='black', label='Learned $f(x)$', alpha=0.75)