#!/usr/bin/env python
# -*- coding: utf-8 -*-

# animator.py
# Author: Ravi Joshi
# Date: 2019/07/30

# import modules
import os
import GPy
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import GPy.plotting.Tango as Tango

#plt.rcParams.update({'font.size': 18})

style = 'seaborn'
if style in plt.style.available:
    plt.style.use(style)

fontsize = 12
plt.rcParams.update({'font.size': fontsize,
                     'xtick.labelsize': fontsize,
                     'ytick.labelsize': fontsize,
                     'axes.labelsize': fontsize,
                     'legend.fontsize': fontsize,
                     'axes.titlesize': fontsize + 2,
                     'text.usetex': False})

MAX_WHILL_MOVE = 10
CURSOR_SIZE = 40


class ModelPlayer():
    def __init__(self, model_file, max_points, timer_freq, dim1, dim2, resolution):
        # load mrd model from pickle file
        self.mrd_model = GPy.load(model_file)

        mrd_X = self.mrd_model.X.mean
        self.mrd_point_count = mrd_X.shape[0]
        if self.mrd_point_count > max_points:
            print('Mean contains more samples. Shape: (%d, %d)' % mrd_X.shape)
            downsample_indices = np.random.choice(self.mrd_point_count, size=max_points, replace=False)
            mrd_X = mrd_X[downsample_indices]

        # parameters for doing latent function inference
        self.q_dim = mrd_X.shape[1]
        self.latent_X = np.zeros((1, self.q_dim))

        self.dim1 = dim1
        self.dim2 = dim2
        self.resolution = resolution

        self.mrd_X = mrd_X[:, [self.dim1, self.dim2]]

        title = 'Baxter Whill Movement using MRD'
        fig1 = plt.figure(1, figsize=(5, 4))
        self.plot_latent_space(fig1)

        fig2 = plt.figure(2, figsize=(5, 4))
        self.plot_ard_weights(fig2)

        fig1.tight_layout()
        fig2.tight_layout()


    def spin(self):
        plt.show()


    def plot_ard_weights(self, fig, y_thresh=0.05):
        scales1 = self.mrd_model.Y0.kern.input_sensitivity(summarize=False)
        scales2 = self.mrd_model.Y1.kern.input_sensitivity(summarize=False)

        scales1 /= scales1.max()
        scales2 /= scales2.max()

        options = {'title':'ARD Weights','ylabel':'ARD Weight','labels':['Joint Angles', 'WHILL Movements']}

        ax = fig.add_subplot(111)

        print scales1.shape[0]

        x = np.arange(scales1.shape[0])
        c1 = Tango.colorsHex['mediumBlue']
        c2 = Tango.colorsHex['darkGreen']
        h1 = ax.bar(x, height=scales1, width=0.8, align='center', color=c1, linewidth=1.3)
        h2 = ax.bar(x, height=scales2, width=0.5, align='center', color=c2, linewidth=0.7)
        ax.plot([-1, scales1.shape[0]], [y_thresh, y_thresh], '--', linewidth=3, color=Tango.colorsHex['mediumRed'])

        # setting the bar plot parameters
        ax.set_xlim(-1, scales1.shape[0] - 1)
        ax.tick_params(axis='both')
        ax.set_xticks(range(scales1.shape[0]))
        #ax.set_title(options['title'])
        ax.set_ylabel(options['ylabel'])
        ax.set_xlabel('Latent Dimensions')
        ax.legend([h1, h2], options['labels'], loc='upper right')

    def plot_latent_space(self, fig, plot_inducing=False, plot_variance=True):
        x_min, y_min = self.mrd_X.min(axis=0)
        x_max, y_max = self.mrd_X.max(axis=0)
        x_r, y_r = x_max - x_min, y_max - y_min
        x_min -= 0.1 * x_r
        x_max += 0.1 * x_r
        y_min -= 0.1 * y_r
        y_max += 0.1 * y_r

        ax = fig.add_subplot(111)
        #ax.scatter(self.mrd_X[:, 0], self.mrd_X[:, 1], marker='o', s=50, color='b', alpha=0.8, label='Train')
        ax.plot(self.mrd_X[:, 0], self.mrd_X[:, 1], color='blue', linewidth=3, alpha=0.6, label='Mean')

        if plot_variance:
            def get_variance(x):
                Xtest_full = np.zeros((x.shape[0], self.q_dim))
                Xtest_full[:, [self.dim1, self.dim2]] = x
                _, var = self.mrd_model.predict(np.atleast_2d(Xtest_full))
                var = var[:, :1]
                return -np.log(var)

            x, y = np.mgrid[x_min : x_max : 1j * self.resolution, y_min : y_max : 1j * self.resolution]
            grid_data = np.vstack((x.flatten(), y.flatten())).T
            grid_variance = get_variance(grid_data).reshape((self.resolution, -1))
            ax.imshow(grid_variance.T, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(x_min, x_max, y_min, y_max))

        if plot_inducing:
            Z = self.mrd_model.Z
            ax.scatter(Z[:, self.dim1], Z[:, self.dim2], color='white', s=CURSOR_SIZE, marker='^', alpha=0.6)

        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))

        ax.grid(False)
        ax.set_aspect('auto')
        #ax.legend(loc='upper right')
        ax.set_xlabel('Latent Dimension %i' % self.dim1)
        ax.set_ylabel('Latent Dimension %i' % self.dim2)

        legend = ax.legend(loc='upper right', frameon=True)
        # Put white background color on the legend.
        legend.get_frame().set_facecolor('white')
        #ax.title.set_text('Latent Space Visualization')


def main():
    project_loc = '/home/ravi/ros_ws/src/baxter_whill_movement'
    files_dir = os.path.join(project_loc, 'files')
    model_file = os.path.join(files_dir, 'mrd_model.pkl')

    max_points = 200
    timer_freq = 5.0
    dim1 = 0
    dim2 = 6
    resolution = 50

    player = ModelPlayer(model_file, max_points, timer_freq, dim1, dim2, resolution)
    player.spin()

if __name__ == '__main__':
    main()
