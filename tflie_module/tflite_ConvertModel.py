import tensorflow as tf

from function import *
from tensorflow.keras.models import *
from keras_retinanet.models import load_model

from keras_retinanet.losses import * 
from keras_retinanet.bin import *
from keras_retinanet.callbacks import *
from keras_retinanet.layers import *
from keras_retinanet.models import *
from keras_retinanet.preprocessing import *
from keras_retinanet.utils import *


def convert_model(model_path, save_model_path, model_name):
    
    
    model = load_model(model_path)#, backbone_name='resnet50')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(save_model_path + model_name + '.tflite', 'wb') as f:
        f.write(tflite_model)
        
        
model_path1='final_model/210114_test/retinanet50_NP.h5'
#model_path1='snapshots0/resnet50_csv_50.h5'
save_model_path1='final_model/210114_test_lite/'
model_name1='retinanet50_NP'

convert_model(model_path1, save_model_path1, model_name1)

custom_objects = retinanet.custom_objects.copy() 
custom_objects.update(keras_resnet.custom_objects) 