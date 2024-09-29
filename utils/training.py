import numpy as np
from tqdm.notebook import tqdm

def _update(weights, gradient, lr):
    weights -= lr * gradient

def mse(y, y_pred):
    return ((y - y_pred) ** 2).mean()

def train(X_train, y_train, forward, backward, loss_fn, epochs, lr):
    np.random.seed(42)

    feature_count = len(X_train[0])
    target_count = y_train[0].size

    current_parameters = np.random.randn(target_count, feature_count) * 1e-2

    losses = []
    parameters = []
    gradients = []

    for epoch in tqdm(range(epochs + 1), desc='Epochs'):
        y_pred = forward(X_train, current_parameters)

        loss = loss_fn(y_train, y_pred)
        epoch_gradient = backward(X_train, y_train, y_pred)

        tqdm.write(f'Epoch {epoch}, loss: {loss:.6f}')

        losses.append(loss)
        parameters.append(current_parameters.copy())
        gradients.append(epoch_gradient)

        _update(current_parameters, epoch_gradient, lr)

    return losses, parameters, gradients, current_parameters

def train_neural(X_train, y_train, forward, backward, loss_fn, epochs, lr):
    np.random.seed(42)

    input_size = len(X_train[0])
    hidden_size = 1
    output_size = y_train[0].size

    current_parameters_1 = np.random.randn(hidden_size, input_size) * 1e-2
    current_parameters_2 = np.random.randn(output_size, hidden_size) * 1e-2

    losses = []
    parameters = []
    gradients = []

    for epoch in tqdm(range(epochs + 1), desc='Epochs'):
        hidden, y_pred = forward(X_train, current_parameters_1, current_parameters_2)
        loss = loss_fn(y_train, y_pred)
        epoch_gradient_1, epoch_gradient_2 = backward(X_train, hidden, y_train, y_pred, current_parameters_2)

        tqdm.write(f'Epoch {epoch}, loss: {loss:.6f}')

        losses.append(loss)
        parameters.append([current_parameters_1[0].copy(), current_parameters_2[0].copy()])
        gradients.append([epoch_gradient_1[0], epoch_gradient_2[0]])

        _update(current_parameters_1, epoch_gradient_1, lr)
        _update(current_parameters_2, epoch_gradient_2, lr)

    return losses, parameters, gradients, current_parameters_1, current_parameters_2