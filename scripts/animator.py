#!/usr/bin/env python
# -*- coding: utf-8 -*-

# animator.py
# Author: Ravi Joshi
# Date: 2019/07/30

# import modules
import os
import GPy
import argparse
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#plt.rcParams.update({'font.size': 18})

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, bitrate=18000)


style = 'seaborn'
if style in plt.style.available:
    plt.style.use(style)


model_file = '/home/ravi/ros_ws/src/baxter_whill_movement/files/mrd_model.pkl'
max_points = 200
timer_freq = 5.0
dim1 = 0
dim2 = 1
resolution = 50
manual = False
max_whill_move = 10
cursor_size = 40


counter = 0
whill_move_handle = None
latent_cursor_handle = None
whill_move = None
cursor_color = 'green'

# load mrd model from pickle file
mrd_model = GPy.load(model_file)

mrd_X = mrd_model.X.mean
mrd_point_count = mrd_X.shape[0]
if mrd_point_count > max_points:
    print('Mean contains more samples. Shape: (%d, %d)' % mrd_X.shape)
    downsample_indices = np.random.choice(mrd_point_count, size=max_points, replace=False)
    mrd_X = mrd_X[downsample_indices]

# parameters for doing latent function inference
q_dim = mrd_X.shape[1]
latent_X = np.zeros((1, q_dim))

dim1 = dim1
dim2 = dim2
resolution = resolution

mrd_X = mrd_X[:, [dim1, dim2]]

title = 'Baxter Whill Movement using MRD'
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))


def plot_latent_space(plot_inducing=False, plot_variance=True):
    x_min, y_min = mrd_X.min(axis=0)
    x_max, y_max = mrd_X.max(axis=0)
    x_r, y_r = x_max - x_min, y_max - y_min
    x_min -= 0.1 * x_r
    x_max += 0.1 * x_r
    y_min -= 0.1 * y_r
    y_max += 0.1 * y_r

    #ax1.scatter(mrd_X[:, 0], mrd_X[:, 1], marker='o', s=50, color='b', alpha=0.8, label='Train')
    #ax1.plot(mrd_X[:, 0], mrd_X[:, 1], color='blue', linewidth=2, alpha=0.8, label='Mean')

    if plot_variance:
        def get_variance(x):
            Xtest_full = np.zeros((x.shape[0], q_dim))
            Xtest_full[:, [dim1, dim2]] = x
            _, var = mrd_model.predict(np.atleast_2d(Xtest_full))
            var = var[:, :1]
            return -np.log(var)

        x, y = np.mgrid[x_min : x_max : 1j * resolution, y_min : y_max : 1j * resolution]
        grid_data = np.vstack((x.flatten(), y.flatten())).T
        grid_variance = get_variance(grid_data).reshape((resolution, -1))
        ax1.imshow(grid_variance.T, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(x_min, x_max, y_min, y_max))

    if plot_inducing:
        Z = mrd_model.Z
        ax1.scatter(Z[:, dim1], Z[:, dim2], color='white', s=cursor_size, marker='^', alpha=0.6)

    ax1.set_xlim((x_min, x_max))
    ax1.set_ylim((y_min, y_max))

    ax1.grid(False)
    ax1.set_aspect('auto')
    #ax1.legend(loc='upper right')
    ax1.set_xlabel('Dimension %i' % dim1)
    ax1.set_ylabel('Dimension %i' % dim2)
    #ax1.title.set_text('Latent Space Visualization')


def plot_whill_movement():
    ax2.grid(True)
    ax2.set_ylim((-1, max_whill_move))
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Predicted Movement')
    #ax2.title.set_text('Whill Movement Visualization')


plot_latent_space()
plot_whill_movement()

#latent_cursor_handle, = ax1.plot([], [], marker='o', markersize=10, color='green', markeredgecolor='none', linestyle='', alpha=0.5)
latent_cursor_handle, = ax1.plot([], [], color='green', linewidth=2, alpha=0.5)
whill_move_handle, = ax2.plot([], [], color='green', linewidth=2, alpha=0.5)

fig.canvas.set_window_title(title)
fig.tight_layout()


def update_figure(counter):
    #print 'counter %d' % counter

    cursor = mrd_X[counter]
    time = counter

    # increment the counter
    #counter += 1

    # stop the timer if we have finished the trajectory
    #if counter >= mrd_point_count:
    #    print 'Trajectory finished'
    #    return latent_cursor_handle, whill_move_handle,


    # update the cursor
    new_time = np.concatenate((latent_cursor_handle.get_xdata(), np.array((cursor[0]), ndmin=1)))
    new_movement = np.concatenate((latent_cursor_handle.get_ydata(), np.array((cursor[1]), ndmin=1)))
    latent_cursor_handle.set_xdata(new_time)
    latent_cursor_handle.set_ydata(new_movement)


    # update the latent variable X before prediction
    latent_X[0, dim1] = cursor[0]
    latent_X[0, dim2] = cursor[1]

    joint_angles = mrd_model.predict(latent_X, Yindex=0)
    joint_angles = joint_angles[0][0,:].tolist()

    # whill movement predicted from 2D latent space
    #whill_movement = mrd_model.predict(latent_X, Yindex=1)
    # whill movement predicted from joint angles
    x_predict, _ = mrd_model.Y0.infer_newX(np.array(joint_angles, ndmin=2), optimize=False)
    y_out = mrd_model.predict(x_predict.mean, Yindex=1)
    whill_movement =  np.mean(y_out[0])


    ax2.scatter(time, whill_movement, marker='o', s=cursor_size, color='green', alpha=0.3)

    # update the cursor
    new_time = np.concatenate((whill_move_handle.get_xdata(), np.array((time), ndmin=1)))
    new_movement = np.concatenate((whill_move_handle.get_ydata(), np.array((whill_movement), ndmin=1)))
    whill_move_handle.set_xdata(new_time)
    whill_move_handle.set_ydata(new_movement)

    return latent_cursor_handle, whill_move_handle,

print 'Creating video. Please wait...'
# 5 Hz
#func_ani = animation.FuncAnimation(fig, update_figure, frames=100, fargs=(counter, ), interval=200, blit=True)
func_ani = animation.FuncAnimation(fig, update_figure, frames=np.arange(mrd_point_count), interval=200, blit=True)
file_name = 'latent_space.mp4'

func_ani.save(file_name, dpi=300, writer=writer)
