"""
.. module:: track.py
   :synopsis: This module implements the Track class
"""

# Third-party modules
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy.lib.recfunctions import append_fields

class Track:
    """
    This class is used to store the coordinates of a trajectory and compute different features
    such as x and y displacements, distance, the mean of distances and angle.
    These features are used to predict the state (e.g immobile, slow, fast) at each point using
    a LSTM neural network.

    Parameters
    ----------
    folder_name : pathlib.PosixPath
        Path to the folder containing the MDF file used to generate the Track object
    index : int
        ID of the Track, can be retrieved from the MDF file
    table : numpy.rec.array
        ndarray built from `coord` dictionary that allows field access using attribute names.
        It first contains `x`, `y` and `frame` attributes and will be extended with other
        measurements/features

    Attributes
    ----------
    folder_name : pathlib.PosixPath
        Path to the folder containing the MDF file used to generate the Track object
    id : int
        ID of the Track, can be retrieved from the MDF file
    table : numpy.rec.array
        ndarray built from `coord` dictionary that allows field access using attribute names.
        It first contains `x`, `y` and `frame` attributes and will be extended with other
        measurements/features
    num_frames : int
        Number of frames
    """
    def __init__(self, table, dim, label):
        self.table = table
        self.dim = dim
        self.label = label
        self.num_frames = len(self.table['x'])
        self.msd = None
        self.alpha = None
        self.logc = None

    def compute_displacements(self, order):
        """Compute x and y displacements between point i and point i+order

        Parameters
        ----------
        order : int
            Order used to define the gap chosen between the two points

        Returns
        -------
        tuple
            Tuple of two numpy.arrays containing the computed x and y displacements
        """
        displ = {}
        if self.dim == 1:
            displ['x'] = self.table['x'][order:] - self.table['x'][:-order]
        elif self.dim == 2:
            displ['x'] = self.table['x'][order:] - self.table['x'][:-order]
            displ['y'] = self.table['y'][order:] - self.table['y'][:-order]
        elif self.dim == 3:
            displ['x'] = self.table['x'][order:] - self.table['x'][:-order]
            displ['y'] = self.table['y'][order:] - self.table['y'][:-order]
            displ['z'] = self.table['z'][order:] - self.table['z'][:-order]
        return displ

    def compute_distances(self, order):
        """Compute distance between point i and point i+order

        Parameters
        ----------
        order : int
            Order used to define the gap chosen between the two points

        Returns
        -------
        numpy.array
            Array containing the computed distances
        """
        try: # Displacements already computed
            displ = {}
            displ['x'] = self.table['displ_x_{}'.format(order)][:-order]
            displ['y'] = self.table['displ_y_{}'.format(order)][:-order]
        except ValueError:
            displ = self.compute_displacements(order)
        if self.dim == 1:
            dist = np.sqrt(displ['x']**2)
        elif self.dim == 2:
            dist = np.sqrt(displ['x']**2 + displ['y']**2)
        elif self.dim == 3:
            dist = np.sqrt(displ['x']**2 + displ['y']**2 + displ['z']**2)
        return dist

    def compute_mean_distances(self, order, point=1):
        """Compute a mean of distances (having a define order) between point i-point, point i and
        point i+point

        Parameters
        ----------
        order : int
            Order used for the distance calculation
        point: int
            The average is computed between point i and points i+/-point

        Returns
        -------
        numpy.array
            Array containing the computed mean distances
        """
        try: # Distances already computed
            dist = self.table['dist_{}'.format(order)][:-order]
        except ValueError:
            dist = self.compute_distances(order)
        mean_dist = []
        for i in range(self.num_frames-point):
            start = i - point
            if start < 0:
                start = 0
            end = i + point
            if end > self.num_frames - point:
                end = self.num_frames - point
            # print(dist[start:end+1], end='\n\n')
            mean_dist.append(np.mean(dist[start:end+1]))
        return mean_dist

    def compute_angles(self, order, randomize_0_angle):
        """Compute angle of point i using points i-order and i+order

        Parameters
        ----------
        order : int
            Order used to define the gap chosen between the three points
        randomize_0_angle: bool
            If True, gives a random angle value if current and previosu (or next) points are at
            the same position. If False, angles is 0

        Returns
        -------
        numpy.array
            Array containing the computed angles (in radian)
            An angle value of 0 means a forward displacement, whereas a value of pi means a
            backward displacement
        """
        angle_list = [1e+20]*order # the angle of the first point can not be computed
        for point in range(order, self.num_frames-order):
            previous_point = np.array([self.table['x'][point-order], self.table['y'][point-order]])
            current_point = np.array([self.table['x'][point], self.table['y'][point]])
            next_point = np.array([self.table['x'][point+order], self.table['y'][point+order]])
            cp_min_pp = current_point - previous_point
            np_min_cp = next_point - current_point
            if np.all(cp_min_pp == 0) or np.all(np_min_cp == 0):
                if randomize_0_angle:
                    while True:
                        angle = randint(-np.pi, np.pi)
                        if angle != 0:
                            break
                else:
                    angle = 0
            else:
                alpha1 = np.math.atan2(cp_min_pp[1], cp_min_pp[0])
                alpha2 = np.math.atan2(np_min_cp[1], np_min_cp[0]) + 2*np.pi
                alpha = (alpha2 - alpha1) % (2*np.pi)
                angle = alpha if (alpha <= np.pi) else alpha - (2*np.pi)
            angle_list.append(angle)
        return angle_list

    def compute_all_features(self, randomize_0_angle=True):
        """Compute and store the features in `self.table`.
        Feature computed: X and Y displacemets (order 1), distances (order 1),
        mean of distances (order 1 and 2), angles (order 1)"""
        displ = self.compute_displacements(1)
        self.table = append_fields(self.table, ['displ_x_1', 'displ_y_1'], [displ['x'], displ['y']],
                                   usemask=False, fill_value=1e+20)
        dist_1 = self.compute_distances(1)
        self.table = append_fields(self.table, 'dist_1', dist_1, usemask=False, fill_value=1e+20)
        mean_dist_1 = self.compute_mean_distances(1)
        mean_dist_2 = self.compute_mean_distances(2)
        self.table = append_fields(self.table, ['mean_dist_1', 'mean_dist_2'],
                                   [mean_dist_1, mean_dist_2], usemask=False, fill_value=1e+20)
        angle_1 = self.compute_angles(1, randomize_0_angle=randomize_0_angle)
        self.table = append_fields(self.table, 'angle_1', angle_1, usemask=False, fill_value=1e+20)

    def get_table_dataframe(self):
        """Converts the structure of the `self.table` attribute into a dataframe and fills
        the missing values with NaNs

        Returns
        -------
        DataFrame
            `self.table` attribute as a DataFrame
        """
        table_df = pd.DataFrame(self.table)
        table_df = table_df[(table_df != 999999) & (table_df != 1e+20)]
        return table_df

    def get_feature_array(self):
        """Create a numpy arrays containing the features computed and stored in `self.table`
        The first and last lines are not taken due to NaN values.

        Returns
        -------
        numpy.array
            Features from `self.table` as an array
            This array of features is used as input for the LSTM neural network which predicting
            the state (e.g immobile, slow, fast) at each point
        """
        table_df = pd.DataFrame(self.table[1:-1])
        table_df = table_df.drop(['x', 'y'], axis=1)
        if 'frame' in table_df:
            table_df = table_df.drop(['frame'], axis=1)
        if 'state' in table_df:
            table_df = table_df.drop(['state'], axis=1)
        table_array = table_df.to_numpy().reshape(1, -1, table_df.shape[1])
        return table_array

    def set_state(self, states):
        """Add the state column to the `self.table` attribute

        Parameters
        ----------
        states : list
            List of integers representing the state (e.g 0=immobile, 1=slow, 2=fast) at each point
        """
        self.table = append_fields(self.table, 'state', states, usemask=False, fill_value=1e+20)

    def predict_states(self, model):
        """Predict the state (e.g immobile, slow, fast) at each point of the track using a trained
        LSTM model. The list of predicted states is then stored in `self.table`

        Parameters
        ----------
        model : Keras model
            Trained LSTM model used to predict the state at each point
        """
        predicted_states = model.predict(self.get_feature_array())
        predicted_states = [1e+20] + predicted_states.argmax(axis=2)[0].tolist() + [1e+20]
        self.set_state(predicted_states)

    def plot_coordinates(self):
        """Plot the coordinates of the track object"""

        cmap = plt.cm.get_cmap('rainbow', self.num_frames+1)
        if self.dim == 1:
            plt.scatter(range(self.num_frames), self.table['x'], lw=0.1, s=70,
                        c=range(self.num_frames), cmap=cmap)
            plt.plot(range(self.num_frames), self.table['x'], '-', c='black', alpha=0.5)
            plt.gca().set_aspect('equal')
            plt.ylabel('X')
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="3%", pad=0.1)
            plt.colorbar(label='Frame', cax=cax)
            plt.show()
        elif self.dim == 2:
            plt.scatter(self.table['x'], self.table['y'], lw=0.1, s=70, c=range(self.num_frames),
                        cmap=cmap)
            plt.plot(self.table['x'], self.table['y'], '-', c='black', alpha=0.5)
            plt.gca().set_aspect('equal')
            plt.xlabel('X')
            plt.ylabel('Y')
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="3%", pad=0.1)
            plt.colorbar(label='Frame', cax=cax)
            plt.show()
        elif self.dim == 3:
            plt.figure()
            axs = plt.axes(projection="3d")
            axs.scatter3D(self.table['x'], self.table['y'], self.table['z'], lw=0.1, s=70,
                         c=range(self.num_frames), cmap=cmap)
            axs.plot3D(self.table['x'], self.table['y'], self.table['z'], '-', c='black', alpha=0.5)
            # ax.set_aspect('equal')
            axs.set_xlabel('X')
            axs.set_ylabel('Y')
            axs.set_zlabel('Z')
            plt.show()
