# FEST method - AnDi challenge


Feature Extraction Stack LSTM (FEST) method used for tasks 1 &amp; 2 of the Anomalous Diffusion challenge



## Install the required python libraries

### By creating a conda environement

```
conda env create --file environment.yml
conda activate andi-env
```

### Or manually with pip

```
pip install numpy==1.19 scikit-learn==0.23.2 pandas==1.1.0 tqdm keras==2.4.3 tensorflow-gpu==2.2.0 matplotlib==3.3.0
```

## Generate the training datasets

Run the following script in order to create datasets of track length 50, 200, 400 and 600 for the training of the different models.

```
./generate_training_datasets
```

