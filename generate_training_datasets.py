import os
from andi import andi_datasets

nb_points = {'train': 15000000, 'val': 3000000}


for track_len in [50, 200, 400, 600]:
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