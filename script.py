# %% [markdown]
# # Training DIPZ

# %%
"""
Train dipz with keras
"""
#'Take only this many inputs (with no args %(const)s)'
_h_take_first = 'Take only this many inputs (with no args %(const)s)'

import sys
import getopt
import os
import time
import datetime
import platform

print(f"Python Platform: {platform.platform()}")
print(f"Python {sys.version}")

# local libs
from layers import Sum
from utils import gaussian_loss
from utils import TRANSFORMS
from utils import scale
from utils import renamed
from utils import build_feature
from utils import get_gaussian_loss_prec

# mlearnin libs
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
print(f"Tensor Flow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print("GPU Resources Available:\n\t",gpus)

from tensorflow import keras
print(f"Keras Version: {keras.__version__}")

from keras import backend as K
from keras.layers import (
    Dense, TimeDistributed, Input, Concatenate, Masking
)
from keras.utils.generic_utils import CustomMaskWarning

# the data libs
import h5py
import json

# random python utility libs
from argparse import ArgumentParser
from pathlib import Path
import warnings

# %%
# DATA_FILEPATH = "./data/data.h5"
DATA_FILEPATH = "./data/subset.h5"    

args = sys.argv[1:]

opts, args = getopt.getopt(args, "m:", "model=")
for opt, arg in opts:
    if opt in ['-m', '--model']:
        model_name = arg

CONFIG_FILEPATH = f"./config/{model_name}.json"
OUTPUT_FILEPATH = f"./models/{model_name}"

# TODO: clean up these hardcoded values
MASK_VALUE = 999
MERGED_NODES = [32]*4

# A function to define and gets the config file 
def get_config(config_path):
    with open(config_path) as cfg:
        config = json.load(cfg)
    return dict(
        jetfeatnames=config["jetfeatnames"],
        trackfeatnames=config["trackfeatnames"],
        targetfeatnames=config["targetfeatnames"],
        batch_size=config["batch_size"],
        epoch_size=config["epoch_size"],
        number_epochs=config["number_epochs"],
        learning_rate=config["lr"],
        tracknodes=config['tracknodes'],
        jetnodes=config['jetnodes'],
    )

# A function that defines and gets the neural network model
def get_model(config, mask_value):
    n_track_inputs = len(config['trackfeatnames'])
    track_inputs = Input(shape=(None,n_track_inputs))

    n_jet_inputs = len(config['jetfeatnames'])
    jet_inputs = Input(shape=(n_jet_inputs))

    # add jet layers
    x = jet_inputs
    for nodes in config['jetnodes']:
        x = Dense(units=nodes, activation='relu')(x)
    jet_latent = x

    # add track layers
    x = track_inputs
    x = Masking(mask_value=mask_value)(x)
    for nodes in config['tracknodes']:
        x = TimeDistributed(Dense(nodes, activation='relu'))(x)
    x = Sum()(x)
    track_latent = x

    # merge the layers
    merged = Concatenate()([jet_latent, track_latent])
    # todo: not clear how many additonal processing layers we should
    # add here
    x = merged
    for nodes in MERGED_NODES:
        x = Dense(nodes, activation='relu')(x)
    out_latent = x
    outputs = keras.layers.Dense(units=2)(out_latent)
    model = keras.Model(
        inputs=[jet_inputs, track_inputs],
        outputs=outputs)
    # print the summary
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=gaussian_loss)
    return model

# A function that imports the dataset we will be working on
def get_dataset(h5_filepath, config, mask_value, take_first=False):
    """
    We make some hardcoded transformations to normalize these inputs
    """

    # pt is log transformed
    # Z0 is divided by 50
    # target is divided by 50

    trf = TRANSFORMS
    # identy function to pass through things that aren't listed above
    def ident(x):
        return x

    sl = slice(None,None,None)
    if take_first:
        sl = slice(0,take_first,None)

    with h5py.File(h5_filepath) as h5file:
        # get track array
        [print(item) for item in h5file.items()]
        td = h5file['fs_tracks_simple_ip']
        tfn = config['trackfeatnames']
        # we can pass through NaNs here
        with np.errstate(invalid='ignore'):
            trackstack = [trf.get(x,ident)(td[x,sl,...]) for x in tfn]
        track_array = np.stack(trackstack, axis=2)
        invalid = np.isnan(td['pt',sl])
        track_array[invalid,:] = mask_value

        # get jet array
        jd = h5file['jets']
        jfn = config['jetfeatnames']
        jetstack = [trf.get(x,ident)(jd[x,sl]) for x in jfn]
        jet_array = np.stack(jetstack, axis=1)

        # get targets
        tfn = config['targetfeatnames']
        targetstack = [trf.get(x,ident)(jd[x,sl]) for x in tfn]
        target_array = np.stack(targetstack, axis=1)

    return jet_array, track_array, target_array

# A function that gets the inputs to save them
def get_inputs(jet_feature_names, track_feature_names):
    track_variables = [build_feature(x) for x in track_feature_names]
    jet_variables = [build_feature(x) for x in jet_feature_names]
    return {
        'input_sequences': [
            {
                'name': 'tracks_loose202102NoIpCuts_absD0DescendingSort',
                'variables': track_variables,
            }
        ],
        'inputs': [
            {
                'name': 'btagging',
                'variables': jet_variables
            }
        ],
        'outputs': [
            {
                'labels': ['z','negLogSigma2'],
                'name': 'dipz'
            }
        ]
    }

# A function that saves the model
def save_model(model, output_dir, inputs):
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'architecture.json', 'w') as arch:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=CustomMaskWarning)
            arch.write(model.to_json(indent=2))

    model.save_weights(output_dir / 'weights.h5')

    with open(output_dir / 'inputs.json', 'w') as inputs_file:
        json.dump(inputs, inputs_file, indent=2)

# A function that runs the neural network training and saves the weights
def run(config_filepath, h5_filepath, num_epochs = 10):
    mask_value = MASK_VALUE
    config = get_config(config_filepath)
    model = get_model(config, mask_value=mask_value)
    jet_inputs, track_inputs, targets = get_dataset(h5_filepath, config, mask_value)
    model.fit([jet_inputs, track_inputs], targets, epochs=num_epochs)
    inputs = get_inputs(config['jetfeatnames'], config['trackfeatnames'])
    save_model(model, inputs=inputs, output_dir=Path(OUTPUT_FILEPATH))
BEGIN = time.time()
print(f"[{datetime.datetime.now().strftime(f'%H:%M:%S')}] Start", flush=True)

run(CONFIG_FILEPATH, DATA_FILEPATH)

print(f"[{datetime.datetime.now().strftime(f'%H:%M:%S')}]", end=" ")
print(f"Runtime: {round(time.time() - BEGIN, 3)}s", flush=True)


