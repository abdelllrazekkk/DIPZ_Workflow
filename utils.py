#Utilities to be used in the training code of the DIPZ neural network

import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, TimeDistributed, Input, Concatenate, Masking
)
from keras.utils.generic_utils import CustomMaskWarning

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('h5_inputs', type=Path)
    parser.add_argument('-c', '--config-file', type=Path, required=True)
    parser.add_argument('-o', '--output-dir',
                        type=Path, default=Path('outputs'))
    parser.add_argument('-f', '--take-first', type=int, const=100, nargs='?',
                        help=_h_take_first)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    return parser.parse_args()

# transforms
def scale(scale, altname=None):
    def trf(x):
        return x * scale
    trf.scale = scale
    trf.altname = altname
    return trf

def renamed(new):
    trf = lambda x: x
    trf.scale = 1.0
    trf.altname = new
    return trf

log_named = lambda x: np.log(x)
log_named.scale = 1.0
log_named.altname = 'log_pt'

TRANSFORMS = {
    'pt': log_named,
    'pt_btagJes': log_named,
    'detectorZ0': scale(0.02, "z0RelativeToBeamspot"),
    'primaryVertexDetectorZ': scale(0.02),
    'eta_btagJes': renamed('eta'),
}

def build_feature(name):
    offset = 0.0
    scale = 1.0
    if trf := TRANSFORMS.get(name):
        name = trf.altname or name
        scale = trf.scale
    return {'name': name, 'offset': offset, 'scale': scale}


def gaussian_loss(targ, pred):
    """
    Basic gaussian loss model. Probably not properly normalized
    """
    z = pred[:,0:1]
    q = pred[:,1:2]
    loss = K.log(2*np.pi) - 0.5 * q + 0.5 * K.square(z - targ) * K.exp(q)
    return loss

def get_gaussian_loss_prec(epsilon):
    def gaussian_loss_prec(targ, pred):
        """
        This seems to be more stable than the gaussian loss above
        """
        z = pred[:,0:1]
        prec = K.abs(pred[:,1:2]) + epsilon
        loss = - K.log(prec) + K.square(z - targ) * prec
        return loss
    return gaussian_loss_prec