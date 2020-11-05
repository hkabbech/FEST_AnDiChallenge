"""
.. module:: utils.py
   :synopsis: This module implements main functions of the main script
"""

# Third-party modules
import pickle
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, append_fields
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# Local modules
from src.track import Track

def retrieve_track_list(dataset_type, pars):
    """Retrieve list of Track objects"""
    path = pars[f"{dataset_type}_dataset"]
    if dataset_type == 'test':
        filename = f'{path}/track_list_task{pars["task"]}_dim{pars["dim"]}.p'
    else:
        filename = f'{path}/track_list_dim{pars["dim"]}.p'
    try:
        track_list = pickle.load(open(filename, 'rb'))
    except FileNotFoundError:
        file_track = open(f'{path}/task{pars["task"]}.txt', 'r')
        try:
            file_ref = open(f'{path}/ref{pars["task"]}.txt', 'r')
            refs = file_ref.readlines()
        except FileNotFoundError:
            refs = None
            label = None
        track_list = []
        for ntrk, trk in enumerate(file_track):
            track_dim = int(trk[0])
            if track_dim != pars['dim']:
                continue
            coordinates = trk.split('\n')[0].split(';')[1:]
            coordinates = [float(coord) for coord in coordinates]
            if refs:
                splited_label = refs[ntrk].strip("\n").split(';')
                if pars['task'] == 1:
                    label = float(splited_label[1]) # alpha value
                elif pars['task'] == 2:
                    label = int(float(splited_label[1])) # model class
            if track_dim == 1:
                x_array = np.array(coordinates)
                table = np.rec.array([x_array],
                                     dtype=[('x', '<f8')])
                track = Track(table, track_dim, label)
                displ_1 = track.compute_displacements(1)
                displ_2 = track.compute_displacements(2)
                track.table = append_fields(track.table[['x']],
                                            ['displ_x_1', 'displ_x_2'],
                                            [displ_1['x'], displ_2['x']],
                                            usemask=False, fill_value=1e+20)
            elif track_dim == 2:
                thr = int(len(coordinates)/track_dim)
                x_array = np.array(coordinates[:thr])
                y_array = np.array(coordinates[thr:])
                table = np.rec.array([x_array, y_array],
                                     dtype=[('x', '<f8'), ('y', '<f8')])
                track = Track(table, track_dim, label)
                displ_1 = track.compute_displacements(1)
                dist_1 = track.compute_distances(1)
                mean_dist_1 = track.compute_mean_distances(1)
                mean_dist_2 = track.compute_mean_distances(2)
                angle_1 = track.compute_angles(1, randomize_0_angle=False)
                track.table = append_fields(track.table[['x', 'y']],
                                            ['displ_x_1', 'displ_y_1', 'dist_1', 'mean_dist_1', 'mean_dist_2', 'angle_1'],
                                            [displ_1['x'], displ_1['y'], dist_1, mean_dist_1, mean_dist_2, angle_1],
                                            usemask=False, fill_value=1e+20)
            elif track_dim == 3:
                thr = int(len(coordinates)/track_dim)
                x_array = np.array(coordinates[:thr])
                y_array = np.array(coordinates[thr:thr*2])
                z_array = np.array(coordinates[thr*2:])
                table = np.rec.array([x_array, y_array, z_array],
                                     dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
                track = Track(table, track_dim, label)
                displ_1 = track.compute_displacements(1)
                dist_1 = track.compute_distances(1)
                mean_dist_1 = track.compute_mean_distances(1)
                mean_dist_2 = track.compute_mean_distances(2)
                track.table = append_fields(track.table[['x', 'y', 'z']],
                                            ['displ_x_1', 'displ_y_1', 'displ_z_1', 'dist_1', 'mean_dist_1', 'mean_dist_2'],
                                            [displ_1['x'], displ_1['y'], displ_1['z'], dist_1, mean_dist_1, mean_dist_2],
                                            usemask=False, fill_value=1e+20)
            track_list.append(track)
        with open(filename, 'wb') as file:
            pickle.dump(track_list, file)
    return np.array(track_list)



def extract_features(track_list, pars):
    """Extract features for the training"""
    feature_set = []
    for track in track_list:
        table_array = structured_to_unstructured(track.table[pars['features']][pars['lim'][0]:pars['lim'][1]])
        table_array = np.expand_dims(table_array, 0)
        feature_set.append(table_array)
    return feature_set

def extract_labels(track_list):
    label_set = np.array([track.label for track in track_list])
    return label_set

def test_track(table, pars):
    if (pars['dim'] == 1\
        and abs(table['x'][-1]) < pars['length_threshold'])\
        or (pars['dim'] == 2\
        and abs(table['x'][-1]) < pars['length_threshold']\
        and abs(table['y'][-1]) < pars['length_threshold'])\
        or (pars['dim'] == 3\
        and abs(table['x'][-1]) < pars['length_threshold']\
        and abs(table['y'][-1]) < pars['length_threshold']\
        and abs(table['z'][-1]) < pars['length_threshold']):
        return True
    return False

def prepare_dataset(dataset_type, pars):

    # Retrieve list of Track objects
    track_list = retrieve_track_list(dataset_type, pars)
    # Extract feature values
    feature_set = np.array(extract_features(track_list, pars))
    feature_set = feature_set.reshape(-1, pars['track_len']-2, pars['num_features'])
    # Extract labels
    label_set = extract_labels(track_list)
    if pars['task'] == 2:
        label_set = to_categorical(label_set)
    # Randomize
    shuffler = np.random.permutation(len(label_set))
    track_list = track_list[shuffler]
    feature_set = feature_set[shuffler]
    label_set = label_set[shuffler]
    # Remove big values
    keep = []
    remove_model = []
    for n, track in enumerate(track_list):
        table = track.table[pars['lim'][0]:pars['lim'][1]]
        if test_track(table, pars):
            keep.append(n)
        else:
            remove_model.append(track.label)
    track_list = track_list[keep]
    feature_set = feature_set[keep]
    label_set = label_set[keep]
    print(f'Number of track removed due to big scale {len(remove_model)} (mean: {np.mean(remove_model):.3})')
    return {'track_list': track_list, 'feature': feature_set, 'label': label_set}

def plot_training_curves(history, pars, best_epoch):
    """Plot loss, acc and MAE curves"""
    x_axis = np.arange(1, len(history.history['loss'])+1)
    # Plot loss curve
    plt.plot(x_axis, history.history['loss'], label='Training')
    plt.plot(x_axis, history.history['val_loss'], label='Validation')
    plt.axvline(best_epoch, color='k', linestyle='--', label='best model')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.xlim([x_axis[0], x_axis[-1]])
    plt.xticks(np.arange(x_axis[0], x_axis[-1], 5))
    plt.savefig(f'{pars["save_path"]}/LSTM{pars["track_len"]}_loss_curve.png', bbox_inches='tight')
    plt.close()
    # Plot accuracy curve
    plt.plot(x_axis, history.history['accuracy'], label='Training')
    plt.plot(x_axis, history.history['val_accuracy'], label='Validation')
    plt.axvline(best_epoch, color='k', linestyle='--', label='best model')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.xlim([x_axis[0], x_axis[-1]])
    plt.xticks(np.arange(x_axis[0], x_axis[-1], 5))
    plt.savefig(f'{pars["save_path"]}/LSTM{pars["track_len"]}_acc_curve.png', bbox_inches='tight')
    plt.close()
    # Plot MAE
    plt.plot(x_axis, history.history['MAE'], label='Training')
    plt.plot(x_axis, history.history['val_MAE'], label='Validation')
    plt.axvline(best_epoch, color='k', linestyle='--', label='best model')
    plt.title('Training and validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend(loc='upper right')
    plt.xlim([x_axis[0], x_axis[-1]])
    plt.xticks(np.arange(x_axis[0], x_axis[-1], 5))
    plt.savefig(f'{pars["save_path"]}/LSTM{pars["track_len"]}_MAE_curve.png', bbox_inches='tight')
    plt.close()


def plot_alpha_pred_label(predictions, labels, pars):
    plt.plot(labels, predictions, 'o', ms=1, alpha=0.1)
    plt.plot(labels, labels, 'r', lw=0.5)
    plt.ylabel(r'$\mathrm{\alpha_{pred}}$')
    plt.xlabel(r'$\mathrm{\alpha_{GT}}$')
    plt.gca().set_aspect('equal')
    plt.title(f"Alpha prediction (MAE={np.mean(abs(predictions-labels)):.3})")
    plt.savefig(f'{pars["save_path"]}/alpha_pred_label.png', bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(matrix, f1score, pars):
    """Plot confusion matrix"""
    _, axis = plt.subplots(figsize=(7, 7))
    axis.matshow(matrix)
    for (i, j), z in np.ndenumerate(matrix):
        axis.text(j, i, f'{z:0.3f}', ha='center', va='center', fontsize=12)
    axis.set_xticklabels([' ']+pars['models'], fontsize=12)
    axis.set_yticklabels([' ']+pars['models'], fontsize=12)
    axis.set_xlabel('Predicted class', fontsize=12)
    axis.xaxis.set_ticks_position('bottom')
    plt.title(f'Confusion matrix (F1={f1score})', fontsize=16)
    plt.savefig(f'{pars["save_path"]}/confusion_matrix.png')
    plt.close()
