"""
    TO DO DOCSTRING

"""


import os
import pickle
from contextlib import redirect_stdout
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

from src.utils import *


# With a GPU usage:
CONFIG = ConfigProto()
CONFIG.gpu_options.allow_growth = True
SESSION = InteractiveSession(config=CONFIG)


if __name__ == "__main__":


    # MAIN PARAMETERS
    #################

    PARS = {
        'task': 2,
        'dim': 3,
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
        'training_datasets': f'data/training_datasets/',
        # 'test_dataset': f'data/development_dataset_for_training/',
        'scoring_dataset': f'data/challenge_for_scoring/',
        'save_path': f'results/task{PARS["task"]}_dim{PARS["dim"]}'
    })
    os.makedirs(PARS['save_path'], exist_ok=True)


    ## NEURAL NETWORKS
    ##################

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
        print(MODEL.summary())
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
        print(MODEL.summary())
        with open(f'{PARS["save_path"]}/task2_network_summary.txt', 'w') as file:
            with redirect_stdout(file):
                MODEL.summary()


    ## PREDICTIONS AND SCORING
    ###########################

    TEST_SET = {}
    TEST_SET['track_list'] = retrieve_track_list('test', PARS)
    TEST_SET['feature'] = extract_features(TEST_SET['track_list'], PARS)
    TEST_SET['label'] = extract_labels(TEST_SET['track_list'])


    SCORE_SET = {}
    SCORE_SET['track_list'] = retrieve_track_list('scoring', PARS)
    SCORE_SET['feature'] = extract_features(SCORE_SET['track_list'], PARS)
    SCORE_SET['label'] = extract_labels(SCORE_SET['track_list'])





    ALL_PREDICT_FUNC = {}

    for track_len in PARS['track_len_list']:

        print('#########################################################\n#\t\t\t\t\t\t\t#')
        print(f'#   TRAINING OF THE STACK LSTM WITH TRACK LENGTH = {track_len:<5}#\n#\t\t\t\t\t\t\t#')
        print('#########################################################\n')


        PARS['track_len'] = track_len


        ## PREPARE TRAIN DATASET
        ########################

        print('\nFeature Extraction and preparation of the training and validation datasets...')
        TRAIN_SET = prepare_dataset('training', PARS)
        VAL_SET = prepare_dataset('validation', PARS)


        ## TRAINING OF THE NEURAL NETWORK
        #################################

        CALLBACKS = [
            EarlyStopping(monitor='val_loss', patience=PARS['patience'], min_delta=1e-4, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-9, verbose=1),
            ModelCheckpoint(filepath=f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_best_model.h5', monitor='val_loss',
                            save_best_only=True, verbose=1)
        ]

        HISTORY = MODEL.fit(x=TRAIN_SET['feature'],
                            y=TRAIN_SET['label'],
                            epochs=PARS['max_epochs'],
                            callbacks=CALLBACKS,
                            batch_size=PARS['batch_size'],
                            shuffle=True,
                            validation_data=(VAL_SET['feature'], VAL_SET['label']),
                            verbose=2)

        BEST_EPOCH = len(HISTORY.history['loss']) - PARS['patience']
        plot_training_curves(HISTORY, PARS, best_epoch=BEST_EPOCH)
        MODEL.save(f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_last_model.h5')
        with open(f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_history.p', 'wb') as file:
            pickle.dump(HISTORY.history, file)



        MODEL = load_model(f'{PARS["save_path"]}/LSTM{track_len}/LSTM{track_len}_best_model.h5')
        @tf.function(experimental_relax_shapes=True)
        def predict(array):
            """
                Prediction with the trained model and GPU usage
            """
            return MODEL(array)

        ## PREDICTION ON THE TEST DATASET
        #################################

        # if PARS['task'] == 1:

        #     PRED = []
        #     remove_track = []
        #     with tqdm(total=len(TEST_SET['track_list'])) as pbar:
        #         for N, track in enumerate(TEST_SET['track_list']):
        #             table = track.table[PARS['lim'][0]:PARS['lim'][1]]
        #             if test_track(table, PARS):
        #                 PRED.append(float(predict(TEST_SET['feature'][N])))
        #             else:
        #                 PRED.append(2) # add alpha = 2
        #                 remove_track.append(track.label['model'])
        #             pbar.update(1)
        #     print(f'Number of tracks removed: {len(remove_track)} (mean alpha {np.mean(remove_track)})')

        #     PRED = np.array(PRED)
        #     MAE = np.mean(abs(PRED - TEST_SET['label']))
        #     print(f'MAE = {MAE:.5}')
        #     plot_alpha_pred_label(PRED, TEST_SET['label'], PARS)

        # elif PARS['task'] == 2:

        #     PRED = []
        #     remove_track = []
        #     with tqdm(total=len(TEST_SET['track_list'])) as pbar:
        #         for N, track in enumerate(TEST_SET['track_list']):
        #             table = track.table[PARS['lim'][0]:PARS['lim'][1]]
        #             if test_track(table, PARS):
        #                 PRED.append(np.argmax(predict(TEST_SET['feature'][N])))
        #             else:
        #                 PRED.append(3) # add model 3
        #                 remove_track.append(track.label['model'])
        #             pbar.update(1)
        #     print(f'Number of tracks removed: {len(remove_track)} (mean model {np.mean(remove_track)})')

        #     F1_SCORE = f1_score(PRED, TEST_SET['label'], average='micro')
        #     print(f'F1 = {F1_SCORE}')
        #     CONF_MATRIX = confusion_matrix(PRED, TEST_SET['label'])
        #     CONF_MATRIX = CONF_MATRIX/np.sum(to_categorical(TEST_SET['label']), axis=0)*100
        #     plot_confusion_matrix(CONF_MATRIX, 'best-model', F1_SCORE, PARS)


        ALL_PREDICT_FUNC[f'LSTM{track_len}'] = predict


    ## COMBINATION OF ALL MODELS, PREDICTION ON THE TEST AND SCORING DATASETS
    #########################################################################


    if PARS['task'] == 1:
        # PRED_TEST = []
        # with tqdm(total=len(TEST_SET['track_list'])) as pbar:
        #     for N, track in enumerate(TEST_SET['track_list']):
        #         table = track.table[PARS['lim'][0]:PARS['lim'][1]]
        #         if test_track(table, PARS):
        #             if track.num_frames <= 100:
        #                 PRED_TEST.append(float(ALL_PREDICT_FUNC['LSTM50'](TEST_SET['feature'][N])))
        #             elif track.num_frames > 100 and track.num_frames <= 300:
        #                 PRED_TEST.append(float(ALL_PREDICT_FUNC['LSTM200'](TEST_SET['feature'][N])))
        #             elif track.num_frames > 300 and track.num_frames <= 500:
        #                 PRED_TEST.append(float(ALL_PREDICT_FUNC['LSTM400'](TEST_SET['feature'][N])))
        #             elif track.num_frames > 500:
        #                 PRED_TEST.append(float(ALL_PREDICT_FUNC['LSTM600'](TEST_SET['feature'][N])))
        #         else:
        #             PRED_TEST.append(2)
        #         pbar.update(1)

        PRED_SCORING = []
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
                    PRED_TEST.append(2)
                pbar.update(1)

        with open(f'{PARS["save_path"]}/task{PARS["task"]}_dim{PARS["dim"]}_challenge.txt', 'w') as file:
            for pred in PRED_SCORING:
                file.write(f'{PARS["dim"]};{pred}\n')


    elif PARS['task'] == 2:
        # PRED_TEST = []
        # with tqdm(total=len(TEST_SET['track_list'])) as pbar:
        #     for N, track in enumerate(TEST_SET['track_list']):
        #         table = track.table[PARS['lim'][0]:PARS['lim'][1]]
        #         if test_track(table, PARS):
        #             if track.num_frames <= 100:
        #                 PRED_TEST.append(ALL_PREDICT_FUNC['LSTM50'](TEST_SET['feature'][N]))
        #             elif track.num_frames > 100 and track.num_frames <= 300:
        #                 PRED_TEST.append(ALL_PREDICT_FUNC['LSTM200'](TEST_SET['feature'][N]))
        #             elif track.num_frames > 300 and track.num_frames <= 500:
        #                 PRED_TEST.append(ALL_PREDICT_FUNC['LSTM400'](TEST_SET['feature'][N]))
        #             elif track.num_frames > 500:
        #                 PRED_TEST.append(ALL_PREDICT_FUNC['LSTM600'](TEST_SET['feature'][N]))
        #         else:
        #             PRED_TEST.append([0, 0, 0, 1, 0])
        #         pbar.update(1)

        PRED_SCORING = []
        with tqdm(total=len(SCORE_SET['track_list'])) as pbar:
            for n, track in enumerate(SCORE_SET['track_list']):
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
