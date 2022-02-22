import tensorflow as tf

from tensorflow.keras.utils import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *

from tensorflow.keras import backend as K

################################################################################################################
# Model Loss function
################################################################################################################

def dice_coef(y_true, y_pred, smooth=0.001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


get_custom_objects().update({
   
    'dice_coef' : dice_coef,
    'dice_coef_loss' : dice_coef_loss,
        
})