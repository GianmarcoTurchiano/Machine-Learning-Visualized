import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from IPython import display
from PIL import Image
import cv2
from io import BytesIO

def plot_2D_loss(ax, parameters, gradients, losses, epoch, arrow_color):
    weights = parameters[:, :, 0].flatten()
    epoch_loss = losses[epoch]
    epoch_gradient = gradients[epoch]
    epoch_weight = weights[epoch]
    dw = epoch_gradient[0]

    ax.grid()

    ax.plot(weights, losses, '--', color='gainsboro', zorder=1)

    ax.quiver(epoch_weight, epoch_loss, -dw, 0, angles='xy', scale_units='xy', color=arrow_color, label='Steepest descent', zorder=2)

    ax.plot(weights[: epoch + 1], losses[: epoch + 1], marker='o', markersize=3, label='Epochs', color='seagreen', zorder=3)

def plot_3D_loss(ax, parameters, gradients, losses, epoch, arrow_color):
    weights_1 = parameters[:, :, 0].flatten()
    weights_2 = parameters[:, :, 1].flatten()
    
    epoch_gradient = gradients[epoch]
    epoch_weight = weights_1[epoch]
    epoch_bias = weights_2[epoch]
    epoch_loss = losses[epoch]

    [dw, db] = epoch_gradient

    ax.plot(weights_1, weights_2, losses, '--', color='silver', zorder=1)

    ax.quiver(epoch_weight, epoch_bias, epoch_loss, -dw, -db, 0, length=1, linewidth=2, color=arrow_color, normalize=True, label='Steepest descent')

    ax.plot(weights_1[: epoch +1], weights_2[: epoch +1], losses[: epoch +1], marker='o', markersize=3, label='Epochs', color='seagreen', zorder=3)

def plot_training(fig, ax_model, ax_loss, losses, parameters, gradients, X_train, y_train, X_test, y_test, plot_model, plot_loss, plot_text, plot_other):
    cmap = mpl.colormaps.get_cmap('coolwarm')

    parameters = np.array(parameters)
    gradient_norms = list(map(np.linalg.norm, gradients))

    min_norm = min(gradient_norms)
    max_norm = max(gradient_norms)

    gamma = 0.5  # lower values will have more space in the color gradient
    color_norm = mpl.colors.PowerNorm(gamma=gamma, vmin=min_norm, vmax=max_norm)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)

    cbar = plt.colorbar(sm, ax=ax_loss)
    cbar.set_label('Gradient Norm (Steepness)')

    ax_loss.tick_params(axis='both', which='major', labelsize=8)

    dh = display.display(fig, display_id=True)

    frame_buffers = []
    
    text_w1 = fig.text(
        0.95, 0.63,
        '',
        fontsize=10,
        horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
    )

    text_w2 = fig.text(
        0.95, 0.6,
        '',
        fontsize=10,
        horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
    )

    text_fn = fig.text(
        0.95, 0.57,
        '',
        fontsize=10,
        horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
    )

    text_loss = fig.text(
        0.95, 0.54,
        '',
        fontsize=10,
        horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
    )

    sorted_indices = np.argsort(X_train[:, 0])

    X_train = X_train[sorted_indices]
    y_train = y_train[sorted_indices]

    for epoch, _ in enumerate(losses):
        ax_model.clear()

        ax_model.set_title(f'Epoch {epoch}/{len(losses) - 1}')

        plot_model(ax_model, X_train, y_train, X_test, y_test, parameters[epoch])

        ax_loss.clear()

        epoch_gradient_norm = gradient_norms[epoch]
        
        arrow_color = cmap(color_norm(epoch_gradient_norm))

        plot_loss(ax_loss, parameters, gradients, losses, epoch, arrow_color)

        epoch_parameters = parameters[epoch][0]
        epoch_loss = losses[epoch]

        plot_text(epoch_parameters, epoch_loss, text_w1, text_w2, text_fn, text_loss)

        plot_other()

        fig.tight_layout()

        dh.update(fig)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        frame_buffers.append(buf)
    
    plt.close()

    return frame_buffers

def save_animation(buffers, experiment_name):
    folder_name = r'animations'
    
    frames = list(map(lambda buf: Image.open(buf), buffers))

    _save_gif(frames, experiment_name, folder_name)
    _save_avi(frames, experiment_name, folder_name)

    map(lambda buf: buf.close(), buffers)

def _save_gif(frames, experiment_name, folder_name):
    frames[0].save(f'{folder_name}/train_{experiment_name}.gif',
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0)
    
def _save_avi(frames, experiment_name, folder_name):
    frames_cv = list(map(lambda frame: cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR), frames))

    height, width, layers = frames_cv[0].shape

    video = cv2.VideoWriter(f'{folder_name}/train_{experiment_name}.avi', cv2.VideoWriter_fourcc(*'DIVX'), layers, (width, height))

    for frame in frames_cv:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()