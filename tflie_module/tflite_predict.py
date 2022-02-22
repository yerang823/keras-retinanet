import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
import skimage.transform import resize


from keras.models import *
from keras import backend as K


from function import *


def predict(model_path, load_dcm_path, save_image_path):
    
    ################################################################################################################
    # Load Image
    ################################################################################################################

    image = sitk.ReadImage(load_dcm_path)
    
    image_array = sitk.GetArrayFromImage(image)
    
    if image_array.shape[0] == 1:
        
        image_array = image_array[0][:,:,0]
        
        image_array = np.expand_dims(image_array, axis=2)
        
    else:
        
        image_array = image_array.transpose(1,2,0)
        
    
    resize_image_array = resize(image_array, (512,512,1))
    
    
    test_image = resize_image_array / 255
    
    ################################################################################################################
    # Load Model
    ################################################################################################################   
    
    model = tf.lite.Interpreter(model_path)
    
    model.allocate_tensors()
    
    input_details = model_1.get_input_details()
    
    output_details = model_1.get_output_details()
        
    model.set_tensor(input_details[0]['index'], test_image)
    
    model.invoke()

    ################################################################################################################
    # Predcit Image
    ################################################################################################################
    
    pred_probability = model.get_tensor(output_details[0]['index'])
    
    output_data = (pred_probability > 0.5) * 1
    
    
    return output_data