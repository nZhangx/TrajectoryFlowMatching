import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_path_ind(traj, groundtruth, t_span=torch.linspace(0, 4 * np.pi, 100), title=""):
    n = len(t_span)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    len_traj = traj.shape[0]
    ax1.scatter([0] * len_traj, traj[0, 0], traj[0, 1], alpha=0.5, c="red")  # start
    for i in range(n - 1):
        ax1.plot([t_span[i], t_span[i + 1]], [traj[i, 0], traj[i + 1, 0]], [traj[i, 1], traj[i + 1, 1]], alpha=1, c="olive")  # path
        ax1.plot([t_span[i], t_span[i + 1]], [groundtruth[i, 0], groundtruth[i + 1, 0]], [groundtruth[i, 1], groundtruth[i + 1, 1]], alpha=1, c="pink")
    ax1.scatter(t_span, traj[:, 0], traj[:, 1], alpha=0.5, c="blue")  # end
    ax1.scatter(t_span, groundtruth[:, 0], groundtruth[:, 1], alpha=0.5, c="purple")  # ground truth
    ax1.set_title(title)

    return fig


def plot_3d_path_ind_noise(traj, groundtruth, noise, t_span=torch.linspace(0, 4 * np.pi, 100), title=""):
    n = len(t_span)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    noise = noise.cpu().numpy()
    
    len_traj = traj.shape[0]
    ax1.scatter([0] * len_traj, traj[0, 0], traj[0, 1], alpha=0.5, c="red")  # start

    # Plot trajectory and ground truth
    ax1.plot(t_span, traj[:, 0], traj[:, 1], label='Trajectory', c='olive')
    ax1.plot(t_span, groundtruth[:, 0], groundtruth[:, 1], label='Ground Truth', c='pink')
    ax1.scatter(t_span, traj[:, 0], traj[:, 1], alpha=0.5, c="blue")  # end
    ax1.scatter(t_span, groundtruth[:, 0], groundtruth[:, 1], alpha=0.5, c="purple")  # ground truth
    # Plot uncertainty as scatter points around each trajectory point
    # Plus and minus noise values for visualization
    for i in range(n-1):
        if i == 0:
            continue
        x_noise_pos = traj[i+1, 0] + noise[i, 0]
        y_noise_pos = traj[i+1, 1] + noise[i, 1]
        x_noise_neg = traj[i+1, 0] - noise[i, 0]
        y_noise_neg = traj[i+1, 1] - noise[i, 1]
        ax1.scatter([t_span[i]]*2, [x_noise_pos, x_noise_neg], [y_noise_pos, y_noise_neg], color='gray', alpha=0.5)

    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X')
    ax1.set_zlabel('Y')
    ax1.legend()

    return fig



def join_3d_plots(figs, rows, cols):
    new_fig = plt.figure(figsize=(15 * cols, 10 * rows))
    for i, fig in enumerate(figs):
        ax = new_fig.add_subplot(rows, cols, i + 1, projection='3d')
        original_ax = fig.get_children()[1]
        for line in original_ax.get_lines():
            ax.add_line(line)
        for patch in original_ax.get_patches():
            ax.add_patch(patch)
    return new_fig