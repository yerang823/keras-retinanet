# -*- coding: utf-8 -*-

import os 
import numpy as np
from sklearn.model_selection import StratifiedKFold
#from sklearn import model_selection
import cv2

splits_number = 5

path="./data/image/img_all/"
normal_image_list=[]
#atypical_image_list=[]
positive_image_list=[]
'''
for dirName, subdirList, fileList in os.walk(path+"img_all/"):
    for i,filename in enumerate(fileList):
        #if ".jpg" in filename.lower():
        if filename[-4:]==".jpg":
            if filename[0]=='N':
                normal_image_list.append(os.path.join(dirName,filename))
                print("filename=",filename)
            elif filename[0]=='A':
                atypical_image_list.append(os.path.join(dirName,filename))
            elif filename[0]=='P':
                positive_image_list.append(os.path.join(dirName,filename))
'''

for dirName, subdirList, fileList in os.walk(path+"NA/"):
    for i,filename in enumerate(fileList):
        if ".jpg" in filename.lower():
            normal_image_list.append(os.path.join(dirName,filename))
'''
for dirName, subdirList, fileList in os.walk(path+"P/"):
    for i,filename in enumerate(fileList):
        if ".jpg" in filename.lower():
            atypical_image_list.append(os.path.join(dirName,filename))
'''
for dirName, subdirList, fileList in os.walk(path+"P/"):
    for i,filename in enumerate(fileList):
        if ".jpg" in filename.lower():
            positive_image_list.append(os.path.join(dirName,filename))

print(len(normal_image_list))
#print(len(atypical_image_list))
print(len(positive_image_list))

all_case=normal_image_list+positive_image_list
all_case_arr_X=np.array(all_case)
all_case_arr_Y=all_case_arr_X.copy()
all_case_arr_Y[:len(normal_image_list)]= 0
all_case_arr_Y[len(normal_image_list):]= 1
#all_case_arr_Y[len(normal_image_list)+len(atypical_image_list) : ]= 2

#kf = StratifiedKFold(all_case_arr_Y, n_splits=splits_number)
kf = StratifiedKFold(n_splits=splits_number)


count=0
for train_index, test_index in kf.split(all_case_arr_X,all_case_arr_Y):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    
    train_x, test_x = all_case_arr_X[train_index], all_case_arr_X[test_index]
    train_y, test_y = all_case_arr_Y[train_index], all_case_arr_Y[test_index]
    
    f=open("./data/image/%s/txt/train.txt"%(count),"w")
    for i,train_x_img in enumerate(train_x):
        img=cv2.imread(train_x_img)
        filename=train_x_img.split("/")[-1]
        if train_y[i] == '0':
            clas='normal'
        #elif train_y[i] == '1':
        #    clas='atypical'
        elif train_y[i] == '1':
            clas='positive'
        #cv2.imwrite("./data/image/%s/train/%s/%s"%(count,clas,filename),img)
        cv2.imwrite("./data/image/%s/train/%s"%(count,filename),img) 
        f.write(train_x_img+'\n')
        #print("train=====", i,"/",len(train_x))
    f.close()
    
    ff=open("./data/image/%s/txt/test.txt"%(count),"w")
    for i,test_x_img in enumerate(test_x):
        img=cv2.imread(test_x_img)
        filename=test_x_img.split("/")[-1]
        if test_y[i] == '0':
            clas='normal'
        #elif test_y[i] == '1':
        #    clas='atypical'
        elif test_y[i] == '1':
            clas='positive'
        #cv2.imwrite("./data/image/%s/test/%s/%s"%(count,clas,filename),img)
        cv2.imwrite("./data/image/%s/test/%s"%(count,filename),img)
        ff.write(test_x_img+'\n')
        #print("test=====", i,"/",len(test_x))
    ff.close()
        
    count=count+1







