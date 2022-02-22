# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:09:15 2018

@author: gachon
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import average_precision_score, precision_recall_curve
import cv2
import pandas as pd
import matplotlib.font_manager as fm
import pandas as pd


num=0


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])  
    
    if xB-xA > 0 and yB-yA > 0:
        interArea = (xB - xA + 1) * (yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou=interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0
    
    return iou
###################################################################
    
#for key in plt.rcParamsDefault.keys():
#    if 'font' in key:
#        print("{}: {}".format(key, plt.rcParamsDefault[key]))
##font.family: ['sans-serif']
#
#font_path = 'C:/WINDOWS/Fonts/arial.ttf'
#font_name = fm.FontProperties(fname=font_path).get_name()
#plt.rcParams['font.family'] = font_name

###################################################################


#number_list=['0','1','2','3','4','5','6']

result=pd.DataFrame(columns=['TP','FP','FN','sensitivity','FP/image'])	
TP_sum=0
FP_sum=0
image_number_sum=0
roi_number_sum=0
FROC=pd.DataFrame(columns=['prob','TP','FP','sensitivity','image/FP'])
TP_all_prob=[]
FP_all_prob=[]
#for test_num in number_list:
f = open("./data/image/%s/txt/test.txt"%str(num), "r")
t_roi=f.readlines()
f.close()

test_image_all_list=[i.split(",")[0] for i in t_roi]
test_image_all_list=list(set(test_image_all_list))

test_image_name_list=[i.split(",")[0] for i in t_roi if not ',,,,,' in i]
test_image_name_list=list(set(test_image_name_list))

image_number=len(test_image_name_list)
image_number_sum=image_number_sum+image_number

#f=open("./result/txt/detection_result_%s.txt"%(test_num),"r")
#p_roi=f.readlines()
#f.close()

#p_roi=pd.read_csv("./detection_result.csv")
f = open("./result/%s/txt/test_pred.txt"%str(num), "r")
p_roi=f.readlines()
f.close()


test_img=[i.split(",")[0] for i in t_roi if not ',,,,,' in i]
roi_number=len(test_img)
roi_number_sum=roi_number_sum+roi_number

normal_FP=0

#pp_list=p_roi['filename']
pp_list=p_roi
for normal_mass in test_image_all_list:
    if not normal_mass in test_image_name_list:
        if normal_mass in pp_list:
            normal_FP=normal_FP+len(p_roi[p_roi['filename']==normal_mass])
#image_path="D:/psjproject/MG/data/all_img/mass/test_img_%s/"%str(resize_img)


TP_all=0
FP_all=0
FN_all=0

for filename in test_image_name_list:
    TP=0
    FP=0
    FN=0
    TP_prob=[]
    FP_prob=[]
    dul_TP_prob=[]
    dul_TP=0
    t_roi_list=[]
    for true_roi in t_roi:
        (file_,x1,y1,x2,y2,_)=true_roi.split(",")
        if file_==filename:
            t_roi_list.append([int(x1), int(y1), int(x2), int(y2)])

    pred_roi_list=[]
    for n,pred_roi in enumerate(p_roi['filename']):
        if filename==pred_roi:
            if p_roi['prob'][n] >= 0.01:
                pred_roi_list.append([int(p_roi['x1'][n]), int(p_roi['y1'][n]),int(p_roi['x2'][n]), int(p_roi['y2'][n]),float(p_roi['prob'][n])])
    #roi_number=len(pred_roi_list)	    
    #roi_number_sum=roi_number_sum+roi_number
    for tt in t_roi_list:
        count=0
        for pp in pred_roi_list:
            iou=get_iou(tt,pp[:-1])
            if iou >= 0.3 and count == 0:
                TP=TP+1
                TP_prob.append(pp)
                count=count+1
            elif iou >= 0.3 and count != 0:
                dul_TP=dul_TP+1
                dul_TP_prob.append(pp)


    FN=len(t_roi_list)-TP
    FP=len(pred_roi_list)-TP-dul_TP
    for pred_roi in pred_roi_list:
        if not pred_roi in TP_prob:
            if not pred_roi in dul_TP_prob:
                FP_prob.append(pred_roi)

    TP_all=TP_all+TP
    FP_all=FP_all+FP
    FN_all=FN_all+FN
    TP_all_prob.append(TP_prob)
    FP_all_prob.append(FP_prob)
#print(pred_name) 
#print("TP:"+str(TP_all))
#print("FP:"+str(FP_all+normal_FP))
#print("FN:"+str(FN_all))
TP_sum=TP_sum+TP_all
FP_sum=FP_sum+FP_all+normal_FP

#print(len(TP_all_prob))
#print(len(FP_all_prob))


#print(str(image_number_sum))
#print(str(int(roi_number_sum)-TP_sum))
#print(len(TP_all_prob))
#print(len(FP_all_prob))
for prob in range(0,100,1):
	TP=0
	FP=0
	for TPP in TP_all_prob:
		for TPPP in TPP:
			if prob <= TPPP[-1]*100:
				TP=TP+1
	for FPP in FP_all_prob:
		for FPPP in FPP:
			if prob <= FPPP[-1]*100:
				FP=FP+1
	#print(TP)
	#print(TP)
	#print(str(int(roi_number_sum)))
	imgFP=float(FP/float(image_number_sum))
	sensitivity=float(TP/float(roi_number_sum))
	#print(str(imgFP))
	#print(str(sensitivity))
	#print(sensitivity)
	FROC=FROC.append({'prob':prob,'TP':TP,'FP':FP,'sensitivity':sensitivity,'image/FP':imgFP},ignore_index=True)
	FROC.to_csv("../result/%s/etc/result_froc.csv"%str(num),index=None)
print("FP/image: "+str(FP_sum/float(image_number_sum)))
print("sensitivity: "+str(TP_sum/float(roi_number_sum)))
result=result.append({'TP':str(TP_sum),'FP':str(FP_sum),'FN':str(int(roi_number_sum)-TP_sum),'sensitivity':str(TP_sum/float(roi_number_sum)),'FP/image':str(FP_sum/float(image_number_sum))},ignore_index=True)
result.to_csv("../result/%s/etc/result.csv"%str(num),index=None)

##FROC.to_csv("D:/psjproject/MG/result/froc/a2.csv",index=None)
fig = plt.figure()
plt.xlabel('False Positives/Image', fontsize=12)
plt.ylabel('Sensitivity', fontsize=12)  
fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
plt.plot(FROC['image/FP'], FROC['sensitivity'], '-', color='#000000')    
plt.axis([0,1,0,1])
plt.savefig("../result/%s/etcfroc.tif"%str(num),dpi=300)
plt.show()


'''
##TF=pd.read_csv("D:/psjproject/MG/result/froc/1000_preprocess_test.csv")
#TF=pd.read_csv("D:/psjproject/MG/result/froc/a2.csv")
#RI=pd.read_csv("D:/psjproject/MG/result/froc/1000_no_TF_color_test.csv")
#fig = plt.figure()
#plt.xlabel('False Positives / Image', fontsize=12)
#plt.ylabel('Sensitivity', fontsize=12)  
#fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
#plt.plot(TF['image/FP'], TF['sensitivity'], '-', color='b',label='Transfer Learning')    
#plt.plot(RI['image/FP'], RI['sensitivity'], '-', color='r',label='Random initial value')   
#plt.axis([0,1.8,0,0.9])
#plt.legend(loc="lower right")
#plt.show()    
'''

   


















