"""

    FEST method for tasks 1 & 2 of the Anomalous Diffusion challenge.

    Usage:
        ./fest.py TASK DIM

    Arguments:
        TASK                                  Task number. Should be 1 or 2.
        DIM                                   Dimension number. Should be 1, 2 or 3

    Options:
        -h, --help                            Show this
"""

__authors__ = "Hélène Kabbech"


# Third-party modules
import os
import sys
import pickle
from contextlib import redirect_stdout
from docopt import docopt
from schema import Schema, And, Use, SchemaError
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from tqdm import tqdm

# Local modules
from src.utils import retrieve_track_list, extract_features, extract_labels, test_track,\
                      prepare_dataset, plot_training_curves, plot_alpha_pred_label,\
                      plot_confusion_matrix


# With a GPU usage:
CONFIG = ConfigProto()
CONFIG.gpu_options.allow_growth = True
SESSION = InteractiveSession(config=CONFIG)


if __name__ == "__main__":


    # MAIN PARAMETERS
    #################

    ARGS = docopt(__doc__, version='fest 1.0')
    SCHEMA = Schema({
        'TASK': And(Use(int), lambda n: 1 <= n <= 2, error='TASK should be 1 or 2.'),
        'DIM': And(Use(int), lambda n: 1 <= n <= 3, error='DIM should be 1, 2 or 3.')
    })
    try:
        SCHEMA.validate(ARGS)
    except SchemaError as err:
        sys.exit(err)

    PARS = {
        'task': int(ARGS['TASK']),
        'dim': int(ARGS['DIM']),
        'track_len_list': [50, 200, 400, 600],
        'nb_points': {'train': 15000000, 'val': 3000000},
        'models': ['ATTM', 'CTRW', 'FBM', 'LW', 'SBM'],
        'length_threshold': 10**4,
        'patience': 20,
        'max_epochs': 1000,
        'batch_size': 2**5
    }
    if PARS['dim'] == 1:
        PARS.update({
            'features': ['displ_x_1', 'displ_x_2'],
            'lim': [0, -2]
        })
    elif PARS['dim'] == 2:
        PARS.update({
            'features': ['displ_x_1', 'displ_y_1', 'dist_1', 'mean_dist_1', 'mean_dist_2', 'angle_1'],
            'lim': [1, -1]
        })
    elif PARS['dim'] == 3:
        PARS.update({
            'features': ['displ_x_1', 'displ_y_1', 'displ_z_1', 'dist_1', 'mean_dist_1', 'mean_dist_2'],
            'lim': [0, -2]
        })

    PARS.update({
        'num_states': len(PARS['models']),
        'num_features': len(PARS['features']),
        'training_datasets': 'data/training_datasets/',
        'test_dataset': 'data/development_dataset_for_training/',
        'scoring_dataset': 'data/challenge_for_scoring/',
        'save_path': f'results/task{PARS["task"]}_dim{PARS["dim"]}'
    })
    os.makedirs(PARS['save_path'], exist_ok=True)


    ## TRAINING OF EACH MODEL
    #########################

    ALL_PREDICT_FUNC = {}

    for track_len in PARS['track_len_list']:
        if not os.path.isfile(f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_last_model.h5'):
            print(f'\n#   TRAINING OF THE STACK LSTM WITH TRACK LENGTH = {track_len:<6}#')

            os.makedirs(f'{PARS["save_path"]}/LSTM{track_len}', exist_ok=True)
            PARS['track_len'] = track_len
            PARS['training_dataset'] = f'{PARS["training_datasets"]}/LSTM{track_len}/training'
            PARS['validation_dataset'] = f'{PARS["training_datasets"]}/LSTM{track_len}/validation'


            ## PREPARATION OF THE TRAINING AND VALIDATION DATASETS
            print('\nFeature Extraction and preparation of the training and validation datasets...')
            TRAIN_SET = prepare_dataset('training', PARS)
            VAL_SET = prepare_dataset('validation', PARS)


            ## NEURAL NETWORK
            if PARS['task'] == 1:
                MODEL = Sequential()
                MODEL.add(Bidirectional(LSTM(2**6, return_sequences=True, dropout=0.1),
                                        input_shape=(None, PARS['num_features']), merge_mode='concat'))
                MODEL.add(Bidirectional(LSTM(2**5, return_sequences=True, dropout=0.1),
                                        input_shape=(None, PARS['num_features']), merge_mode='concat'))
                MODEL.add(Bidirectional(LSTM(2**4, return_sequences=False, dropout=0.1),
                                        input_shape=(None, PARS['num_features']), merge_mode='concat'))
                MODEL.add(Dense(2**5, activation='relu'))
                MODEL.add(Dropout(0.2))
                MODEL.add(Dense(2**4, activation='relu'))
                MODEL.add(Dropout(0.2))
                MODEL.add(Dense(2**3, activation='relu'))
                MODEL.add(Dropout(0.1))
                MODEL.add(Dense(1))
                MODEL.compile(loss='mean_squared_error', optimizer='adam', metrics=['MAE', 'accuracy'])
                MODEL.summary()
                with open(f'{PARS["save_path"]}/task1_network_summary.txt', 'w') as file:
                    with redirect_stdout(file):
                        MODEL.summary()


            elif PARS['task'] == 2:
                MODEL = Sequential()
                MODEL.add(Bidirectional(LSTM(2**7, return_sequences=True, dropout=0.1),
                                        input_shape=(None, PARS['num_features']), merge_mode='concat'))
                MODEL.add(Bidirectional(LSTM(2**6, return_sequences=True, dropout=0.1),
                                        input_shape=(None, PARS['num_features']), merge_mode='concat'))
                MODEL.add(Bidirectional(LSTM(2**5, return_sequences=False, dropout=0.1),
                                        input_shape=(None, PARS['num_features']), merge_mode='concat'))
                MODEL.add(Dense(2**6, activation='tanh'))
                MODEL.add(Dropout(0.2))
                MODEL.add(Dense(2**5, activation='tanh'))
                MODEL.add(Dropout(0.2))
                MODEL.add(Dense(2**4, activation='tanh'))
                MODEL.add(Dropout(0.1))
                MODEL.add(Dense(PARS['num_states'], activation='softmax'))
                MODEL.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'MAE'])
                MODEL.summary()
                with open(f'{PARS["save_path"]}/task2_network_summary.txt', 'w') as file:
                    with redirect_stdout(file):
                        MODEL.summary()

            ## TRAINING
            print('\nTraining of the model...')
            CALLBACKS = [
                EarlyStopping(monitor='val_loss', patience=PARS['patience'], min_delta=1e-4, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-9, verbose=1),
                ModelCheckpoint(filepath=f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_best_model.h5',
                                monitor='val_loss', save_best_only=True, verbose=1)
            ]

            HISTORY = MODEL.fit(x=TRAIN_SET['feature'],
                                y=TRAIN_SET['label'],
                                epochs=PARS['max_epochs'],
                                callbacks=CALLBACKS,
                                batch_size=PARS['batch_size'],
                                shuffle=True,
                                validation_data=(VAL_SET['feature'], VAL_SET['label']),
                                verbose=2)


            ## SAVE
            BEST_EPOCH = len(HISTORY.history['loss']) - PARS['patience']
            plot_training_curves(HISTORY, PARS, best_epoch=BEST_EPOCH)
            MODEL.save(f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_last_model.h5')
            with open(f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_history.p', 'wb') as file:
                pickle.dump(HISTORY.history, file)

        MODEL = load_model(f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_best_model.h5')
        @tf.function(experimental_relax_shapes=True)
        def predict(array):
            """
                Prediction using the trained model and with a GPU usage
            """
            return MODEL(array)

        ALL_PREDICT_FUNC[f'LSTM{track_len}'] = predict


    ## PREDICTION ON TEST AND SCORING DATASETS
    ##########################################

    TEST_SET = {}
    TEST_SET['track_list'] = retrieve_track_list('test', PARS)
    TEST_SET['feature'] = extract_features(TEST_SET['track_list'], PARS)
    TEST_SET['label'] = extract_labels(TEST_SET['track_list'])


    SCORE_SET = {}
    SCORE_SET['track_list'] = retrieve_track_list('scoring', PARS)
    SCORE_SET['feature'] = extract_features(SCORE_SET['track_list'], PARS)
    SCORE_SET['label'] = extract_labels(SCORE_SET['track_list'])


    if PARS['task'] == 1:
        PRED_TEST = []
        REMOVE_TRACK = []
        print('\nPrediction on the test dataset...')
        with tqdm(total=len(TEST_SET['track_list'])) as pbar:
            for N, track in enumerate(TEST_SET['track_list']):
                table = track.table[PARS['lim'][0]:PARS['lim'][1]]
                if test_track(table, PARS):
                    if track.num_frames <= 100:
                        PRED_TEST.append(float(ALL_PREDICT_FUNC['LSTM50'](TEST_SET['feature'][N])))
                    elif track.num_frames > 100 and track.num_frames <= 300:
                        PRED_TEST.append(float(ALL_PREDICT_FUNC['LSTM200'](TEST_SET['feature'][N])))
                    elif track.num_frames > 300 and track.num_frames <= 500:
                        PRED_TEST.append(float(ALL_PREDICT_FUNC['LSTM400'](TEST_SET['feature'][N])))
                    elif track.num_frames > 500:
                        PRED_TEST.append(float(ALL_PREDICT_FUNC['LSTM600'](TEST_SET['feature'][N])))
                else:
                    REMOVE_TRACK.append(track.label)
                    PRED_TEST.append(2)
                pbar.update(1)
            print(f'{len(REMOVE_TRACK)} tracks removed (mean alpha {np.mean(REMOVE_TRACK):.3})')

            PRED_TEST = np.array(PRED_TEST)
            MAE_SCORE = np.mean(abs(PRED_TEST - TEST_SET['label']))
            print(f'MAE = {MAE_SCORE:.3}')
            plot_alpha_pred_label(PRED_TEST, TEST_SET['label'], PARS)

        PRED_SCORING = []
        print('\nPrediction on the scoring dataset...')
        with tqdm(total=len(SCORE_SET['track_list'])) as pbar:
            for N, track in enumerate(SCORE_SET['track_list']):
                table = track.table[PARS['lim'][0]:PARS['lim'][1]]
                if test_track(table, PARS):
                    if track.num_frames <= 100:
                        PRED_SCORING.append(float(ALL_PREDICT_FUNC['LSTM50'](SCORE_SET['feature'][N])))
                    elif track.num_frames > 100 and track.num_frames <= 300:
                        PRED_SCORING.append(float(ALL_PREDICT_FUNC['LSTM200'](SCORE_SET['feature'][N])))
                    elif track.num_frames > 300 and track.num_frames <= 500:
                        PRED_SCORING.append(float(ALL_PREDICT_FUNC['LSTM400'](SCORE_SET['feature'][N])))
                    elif track.num_frames > 500:
                        PRED_SCORING.append(float(ALL_PREDICT_FUNC['LSTM600'](SCORE_SET['feature'][N])))
                else:
                    PRED_SCORING.append(2)
                pbar.update(1)

        with open(f'{PARS["save_path"]}/task{PARS["task"]}_dim{PARS["dim"]}_challenge.txt', 'w') as file:
            for pred in PRED_SCORING:
                file.write(f'{PARS["dim"]};{pred}\n')


    elif PARS['task'] == 2:
        PRED_TEST = []
        REMOVE_TRACK = []
        print('\nPrediction on the test dataset...')
        with tqdm(total=len(TEST_SET['track_list'])) as pbar:
            for N, track in enumerate(TEST_SET['track_list']):
                table = track.table[PARS['lim'][0]:PARS['lim'][1]]
                if test_track(table, PARS):
                    if track.num_frames <= 100:
                        PRED_TEST.append(np.argmax(ALL_PREDICT_FUNC['LSTM50'](TEST_SET['feature'][N])))
                    elif track.num_frames > 100 and track.num_frames <= 300:
                        PRED_TEST.append(np.argmax(ALL_PREDICT_FUNC['LSTM200'](TEST_SET['feature'][N])))
                    elif track.num_frames > 300 and track.num_frames <= 500:
                        PRED_TEST.append(np.argmax(ALL_PREDICT_FUNC['LSTM400'](TEST_SET['feature'][N])))
                    elif track.num_frames > 500:
                        PRED_TEST.append(np.argmax(ALL_PREDICT_FUNC['LSTM600'](TEST_SET['feature'][N])))
                else:
                    REMOVE_TRACK.append(track.label)
                    PRED_TEST.append(3)
                pbar.update(1)
            print(f'{len(REMOVE_TRACK)} tracks removed (mean model {np.mean(REMOVE_TRACK):.3})')

            F1_SCORE = f1_score(PRED_TEST, TEST_SET['label'], average='micro')
            print(f'F1 = {F1_SCORE:.3}')
            CONF_MATRIX = confusion_matrix(PRED_TEST, TEST_SET['label'])
            CONF_MATRIX = CONF_MATRIX/np.sum(to_categorical(TEST_SET['label']), axis=0)*100
            plot_confusion_matrix(CONF_MATRIX, F1_SCORE, PARS)

        PRED_SCORING = []
        print('\nPrediction on the scoring dataset...')
        with tqdm(total=len(SCORE_SET['track_list'])) as pbar:
            for N, track in enumerate(SCORE_SET['track_list']):
                table = track.table[PARS['lim'][0]:PARS['lim'][1]]
                if test_track(table, PARS):
                    if track.num_frames <= 100:
                        PRED_SCORING.append(ALL_PREDICT_FUNC['LSTM50'](SCORE_SET['feature'][N]).numpy()[0].tolist())
                    elif track.num_frames > 100 and track.num_frames <= 300:
                        PRED_SCORING.append(ALL_PREDICT_FUNC['LSTM200'](SCORE_SET['feature'][N]).numpy()[0].tolist())
                    elif track.num_frames > 300 and track.num_frames <= 500:
                        PRED_SCORING.append(ALL_PREDICT_FUNC['LSTM400'](SCORE_SET['feature'][N]).numpy()[0].tolist())
                    elif track.num_frames > 500:
                        PRED_SCORING.append(ALL_PREDICT_FUNC['LSTM600'](SCORE_SET['feature'][N]).numpy()[0].tolist())
                else:
                    PRED_SCORING.append([0, 0, 0, 1, 0])
                pbar.update(1)

        with open(f'{PARS["save_path"]}/task{PARS["task"]}_dim{PARS["dim"]}_challenge.txt', 'w') as file:
            for pred in PRED_SCORING:
                file.write(f'{PARS["dim"]};{";".join(map(str, pred))}\n')
