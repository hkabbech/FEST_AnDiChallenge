"""

    Generate AnDi datasets for the training


    Usage:
        ./generate_training_datasets.py NUM_POINTS

    Arguments:
        NUM_POINTS                            The number of points to get in each dataset

    Options:
        -h, --help                            Show this
"""

import os
import sys
from andi import andi_datasets
from docopt import docopt
from schema import Schema, And, Use, SchemaError


if __name__ == "__main__":

    ARGS = docopt(__doc__, version='fest 1.0')
    SCHEMA = Schema({
        'NUM_POINTS': And(Use(int), lambda n: n >= 10000, error='NUM_POINTS should be >= 10,000.'),
    })
    try:
        SCHEMA.validate(ARGS)
    except SchemaError as err:
        sys.exit(err)

    nb_points = {'train': int(ARGS['NUM_POINTS']), 'val': int(int(ARGS['NUM_POINTS'])*0.2)}

    for track_len in [50, 200, 400, 600]:
        print(f'\ntraining/validation datasets with track length = {track_len}:')
        train_path = f'data/training_datasets/LSTM{track_len}/training/'
        val_path = f'data/training_datasets/LSTM{track_len}/validation/'
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        dataset = andi_datasets().andi_dataset(N=nb_points['train']//track_len, save_dataset=True,
                                     tasks=[1, 2], dimensions=[1, 2, 3],
                                     min_T=track_len, max_T=track_len+1,
                                     path_datasets=train_path)
        os.makedirs(val_path, exist_ok=True)
        dataset = andi_datasets().andi_dataset(N=nb_points['val']//track_len, save_dataset=True,
                                     tasks=[1, 2], dimensions=[1, 2, 3],
                                     min_T=track_len, max_T=track_len+1,
                                     path_datasets=val_path)
