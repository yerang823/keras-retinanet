#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras

import sys
sys.path.insert(0, '../')


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
#gpu = 0

# set the modified tf session as backend in keras
#setup_gpu(gpu)

#num=0

# cv 5
for num in range(5):

  print( "CV = ", num , "=========================================")
  img_save_path='./result/%s/img/'%str(num)
  txt_save_path='./result/%s/txt/'%str(num)
  
  # ## Load RetinaNet model
  
  # In[ ]:
  
  
  # adjust this to point to your downloaded/trained model
  # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
  model_path = os.path.join('./final_model/retinanet50_cv%s_201015.h5'%str(num)) # 'snapshots', 'resnet50_coco_best_v2.1.0.h5') #'resnet50_csv_50.h5')#
  
  # load retinanet model
  model = models.load_model(model_path, backbone_name='resnet50')
  
  # if the model is not converted to an inference model, use the line below
  # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
  # model = models.convert_model(model)
  
  #print(model.summary())
  
  # load label to names mapping for visualization purposes
  '''
  labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
  '''
  
  labels_to_names={0:'normal', 1:'positive'}
  
  # ## Run detection on example
  
  # In[ ]:
  
  sess=tf.Session()
  
  # load image
  f=open('./data/image/%s/txt/test.txt'%str(num),'r') # image to predict
  lines=f.readlines()
  f.close()
  
  ff=open(txt_save_path+'test_pred_201015.txt','w') # predicted image
  for i in range(len(lines)):
      file_path=lines[i].split(',')[0][:-1]
      #file_path=lines[i][:-1]
      file_name=file_path.split('/')[-1]
      image = read_image_bgr(file_path)# '000000008021.jpg')
  
      # copy to draw on
      draw = image.copy()
      draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
      
      # preprocess image for network
      image = preprocess_image(image)
      image, scale = resize_image(image,500,830)#800,1333)
      
      # process image
      start = time.time()
      boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
      
      #print("processing time: ", time.time() - start)
      
      # correct for image scale
      boxes /= scale
      
      
      # visualize detections
      num=tf.argmax(scores[0]).eval(session=sess)
      
      label=labels[0][num]
      box=boxes[0][num]
      score=scores[0][num]
      
      color = label_color(label)
      b = box.astype(int) 
      draw_box(draw, b, color=color)
      
      caption = "{},{:.3f}".format(labels_to_names[label],score)
      draw_caption(draw, b, caption)
      '''
      for box, score, label in zip(boxes[0], scores[0], labels[0]):
          # scores are sorted so we can break
          print("boxes======",boxes[0])
          print("labels======",labels[0])
          if score < 0.5:
              break
              
          color = label_color(label)
          
          b = box.astype(int)
          draw_box(draw, b, color=color)
          
          caption = "{},{:.3f}".format(labels_to_names[label], score)
          draw_caption(draw, b, caption)
      '''
              
      plt.imsave(img_save_path+file_name, draw)
      #ff.write(file_path+','+caption+'\n')
      #ff.write(file_path+','+str(b[0])+','+str(b[1])+','+str(b[2])+','+str(b[3])+','+caption+'\n')
      
      '''
      #temp
      clas= file_path.split(',')[0].split('/')[-1][0]
      if clas=='N':
          class_='noraml'
      elif clas=='A':
          class_='atypical'
      elif clas=='P':
          class_='positive'
      '''
      
      ff.write(file_path+','+str(b[0])+','+str(b[1])+','+str(b[2])+','+str(b[3])+','+labels_to_names[label]+'\n') #caption+'\n')
      print(i,"/",len(lines))
  
  ff.close()


# In[ ]:





# In[ ]:




