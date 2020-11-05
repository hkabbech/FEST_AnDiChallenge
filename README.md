![Python version](https://img.shields.io/badge/python-3-brightgreen.svg)
[![License: GNU](https://img.shields.io/badge/License-GNU-yellow.svg)](http://www.gnu.org/licenses/gpl-3.0.en.html)

<br>

# FEST method - AnDi challenge

The FEST method was implemented to solve task 1 and 2 in all 1, 2 or 3 dimensions from the Anomalous Diffusion challenge.
The task 1 consists in the inference of the anomalous diffusion exponent α while the task 2 is a classification challenge of diffusion models.

As indicated by its name, the principle of the FEST (Feature Extraction Stack LSTM) method is first the measurement at each point of 6 features which vary depending on the dimension; These features could be x, y or z displacements, distances, mean of distances and/or angle. Afterwards, the input of features is passed through a neural network of stack bidirectional LSTM and Dense layers to predict either the alpha exponent or the diffusion model. Because this network is limited to one track length during the training, we decided to train 4 different model each of which having a different track length (50, 200, 400 and 600), finally a combination of all 4 models is used during the prediction of tracks with various length.


## Install the required python libraries

### By creating a conda environement

```
conda env create --file environment.yml
conda activate andi-env
```

### Or manually with pip

```
pip install numpy==1.19 scikit-learn==0.23.2 pandas==1.1.0\
            tqdm keras==2.4.3 tensorflow-gpu==2.2.0\
            matplotlib==3.3.0 docopt schema
```

## Generate the training datasets

Run the following script in order to create AnDi datasets of track length 50, 200, 400 and 600 for the training of the 4 models.

```
./generate_training_datasets
```

The `development_dataset_for_training` and `challenge_for_scoring` datasets have to be placed in `data/` folder.

## Run the script

### Get help

```
Usage:
    ./fest.py TASK DIM

Arguments:
    TASK                                  Task number. Should be 1 or 2.
    DIM                                   Dimension number. Should be 1, 2 or 3

Options:
    -h, --help                            Show this
```

### Toy example

Run the task 1 dimension 3:
```
./fest 1 3
```
