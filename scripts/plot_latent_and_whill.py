#!/usr/bin/env python
# -*- coding: utf-8 -*-

# animator.py
# Author: Ravi Joshi
# Date: 2019/07/30

# import modules
import os
import GPy
import rospy
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator

#plt.rcParams.update({'font.size': 18})

style = 'seaborn'
if style in plt.style.available:
    plt.style.use(style)

fontsize = 16 # previously it was 14
plt.rcParams.update({'font.size': fontsize,
                     'xtick.labelsize': fontsize,
                     'ytick.labelsize': fontsize,
                     'axes.labelsize': fontsize,
                     'legend.fontsize': fontsize,
                     'axes.titlesize': fontsize + 2,
                     'text.usetex': False})

STOP_FRAME = 166

MAX_WHILL_MOVE = 10
CURSOR_SIZE = 40

whill_from_latent = []
whill_from_joint = []

project_loc = '/home/ravi/ros_ws/src/baxter_whill_movement'
files_dir = os.path.join(project_loc, 'files')


def save_data():
    header = 'movement'

    latent_file = os.path.join(files_dir, 'whill_from_latent.csv')
    joint_file =  os.path.join(files_dir, 'whill_from_joint.csv')

    np.savetxt(latent_file, whill_from_latent, delimiter=',', fmt='%.6f', header=header, comments='')
    np.savetxt(joint_file, whill_from_joint, delimiter=',', fmt='%.6f', header=header, comments='')


class ModelPlayer():
    def __init__(self, model_file, max_points, timer_freq, dim1, dim2, resolution, manual):
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
        fig, (self.ax1, self.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        #self.ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.plot_latent_space()
        self.plot_whill_movement()
        fig.canvas.set_window_title(title)
        #self.text_handle = self.ax1.text(0.8, 0.1, 'Play Mode: OFF', horizontalalignment='center', verticalalignment='center', transform=self.ax1.transAxes, bbox={'facecolor':'green', 'alpha':0.5, 'pad':6})

        self.counter = 0
        self.whill_move_handle = None
        self.latent_cursor_handle = None

        if manual:
            #fig.suptitle('Predicted Whill movements are not sent to the Whill controller', fontstyle='italic', color='red')
            # variables for mouse cursor based motion
            self.mouse_xy = np.zeros((1, 2))
            self.start_motion = False
            self.cursor_color = 'red'

            # connect the cursor class
            fig.canvas.mpl_connect('button_press_event',self.mouse_click)
            fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
            fig.subplots_adjust(top=0.80)
        else:
            self.whill_move = None
            #self.init_ros_whillpy()
            self.cursor_color = 'green'
            #self.text_handle.set_text('Automatic Mode: ON')

            # create a timer to follow the mean trajectory
            self.ros_timer = rospy.Timer(rospy.Duration(1 / timer_freq), self.timer_callback)

        # adjust the space at the bottom
        #fig.subplots_adjust(bottom=0.15)
        fig.tight_layout()


    def spin(self):
        plt.show()


    def mouse_click(self, event):
        if not event.inaxes:
            return

        self.start_motion = ~self.start_motion
        '''
        if self.start_motion:
            #self.text_handle.set_text('Play Mode: ON')
            #self.text_handle.set_bbox({'facecolor':'red', 'alpha':0.5, 'pad':10})
        else:
            #self.text_handle.set_text('Play Mode: OFF')
            #self.text_handle.set_bbox({'facecolor':'green', 'alpha':0.5, 'pad':10})
        '''

        self.latent_cursor_handle.axes.figure.canvas.draw_idle()


    def update_whill_plot(self, time, whill_movement):
        self.ax2.scatter(time, whill_movement, marker='o', s=CURSOR_SIZE, color='green', alpha=0.3)
        if self.whill_move_handle is None:
            # initialize the plot handle if it is null
            self.whill_move_handle, = self.ax2.plot(time, whill_movement, color='green', linewidth=2, alpha=0.5)
        else:
            # update the cursor
            new_time = np.concatenate((self.whill_move_handle.get_xdata(), np.array((time), ndmin=1)))
            new_movement = np.concatenate((self.whill_move_handle.get_ydata(), np.array((whill_movement), ndmin=1)))
            self.whill_move_handle.set_xdata(new_time)
            self.whill_move_handle.set_ydata(new_movement)

        self.whill_move_handle.axes.figure.canvas.draw_idle()


    def update_latent_cursor(self, cursor):
        if self.latent_cursor_handle is None:
            self.latent_cursor_handle = self.ax1.scatter(cursor[0, 0], cursor[0, 1], marker='o', s=CURSOR_SIZE, color=self.cursor_color, alpha=0.5)
        else:
            new_offset = np.concatenate((self.latent_cursor_handle.get_offsets(), cursor))
            self.latent_cursor_handle.set_offsets(new_offset)

        self.latent_cursor_handle.axes.figure.canvas.draw_idle()


    def mouse_move(self, event):
        if not event.inaxes:
            return

        # get the current mouse cursor position
        cursor = np.array((event.xdata, event.ydata), ndmin=2)
        if np.linalg.norm(cursor - self.mouse_xy) < 0.05:
            return

        time = self.counter

        # increment the counter
        self.counter += 1

        # store the current mouse position
        self.mouse_xy = cursor.copy()

        self.update_latent_cursor(cursor)
        joint_angles = self.get_joint_angles(cursor)

        whill_movement = self.get_whill_movement(joint_angles)
        self.update_whill_plot(time, whill_movement)


    def timer_callback(self, data):
        cursor = self.mrd_X[self.counter]
        time = self.counter

        # increment the counter
        self.counter += 1

        # stop the timer if we have finished the trajectory
        if self.counter >= self.mrd_point_count:
            self.ros_timer.shutdown()
            rospy.loginfo('Trajectory finished')
            return

        # get the mean value of the current X
        cursor = np.array((cursor[0], cursor[1]), ndmin=2)

        self.update_latent_cursor(cursor)
        joint_angles = self.get_joint_angles(cursor)

        # whill movement predicted from 2D latent space
        whill_movement_latent = self.mrd_model.predict(self.latent_X, Yindex=1)
        # whill movement predicted from joint angles
        whill_movement = self.get_whill_movement(joint_angles)
        self.update_whill_plot(time, whill_movement)

        whill_from_latent.append(np.mean(whill_movement_latent[0]))
        whill_from_joint.append(whill_movement)

        # stop the timer if we have finished the trajectory
        if self.counter >= STOP_FRAME:
            self.ros_timer.shutdown()
            rospy.loginfo('Stop condition arrived')
            return


    def plot_latent_space(self, plot_inducing=False, plot_variance=True):
        x_min, y_min = self.mrd_X.min(axis=0)
        x_max, y_max = self.mrd_X.max(axis=0)
        x_r, y_r = x_max - x_min, y_max - y_min
        x_min -= 0.1 * x_r
        x_max += 0.1 * x_r
        y_min -= 0.1 * y_r
        y_max += 0.1 * y_r

        #self.ax1.scatter(self.mrd_X[:, 0], self.mrd_X[:, 1], marker='o', s=50, color='b', alpha=0.8, label='Train')
        #self.ax1.plot(self.mrd_X[:, 0], self.mrd_X[:, 1], color='blue', linewidth=2, alpha=0.8, label='Mean')

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
            self.ax1.imshow(grid_variance.T, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(x_min, x_max, y_min, y_max))

        if plot_inducing:
            Z = self.mrd_model.Z
            self.ax1.scatter(Z[:, self.dim1], Z[:, self.dim2], color='white', s=CURSOR_SIZE, marker='^', alpha=0.6)

        self.ax1.set_xlim((x_min, x_max))
        self.ax1.set_ylim((y_min, y_max))

        self.ax1.grid(False)
        self.ax1.set_aspect('auto')
        #self.ax1.legend(loc='upper right')
        self.ax1.set_xlabel('Dimension %i' % self.dim1)
        self.ax1.set_ylabel('Dimension %i' % self.dim2)
        #self.ax1.title.set_text('Latent Space Visualization')


    def plot_whill_movement(self):
        self.ax2.grid(True)
        self.ax2.set_ylim((-1, MAX_WHILL_MOVE))
        self.ax2.set_xlabel('Timestamp')
        self.ax2.set_ylabel('Predicted Movement')
        #self.ax2.title.set_text('Whill Movement Visualization')


    def get_joint_angles(self, cursor):
        # update the latent variable X before prediction
        self.latent_X[0, self.dim1] = cursor[0, 0]
        self.latent_X[0, self.dim2] = cursor[0, 1]

        joint_angles = self.mrd_model.predict(self.latent_X, Yindex=0)
        return joint_angles[0][0,:].tolist()


    def get_whill_movement(self, joint_angles):
        x_predict, _ = self.mrd_model.Y0.infer_newX(np.array(joint_angles, ndmin=2), optimize=False)
        y_out = self.mrd_model.predict(x_predict.mean, Yindex=1)
        return np.mean(y_out[0])


def main():
    rospy.init_node('some_random_name')
    rospy.on_shutdown(save_data)

    model_file = os.path.join(files_dir, 'mrd_model.pkl')

    max_points = 200
    timer_freq = 5.0
    dim1 = 0
    dim2 = 1
    resolution = 50
    manual = False

    player = ModelPlayer(model_file, max_points, timer_freq, dim1, dim2, resolution, manual)
    player.spin()

if __name__ == '__main__':
    main()
