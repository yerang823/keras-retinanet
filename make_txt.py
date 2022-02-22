import os
import cv2
from read_roi import read_roi_file

num=0

train_path="./data/image/%d/train/"%str(num)
test_path="./data/image/%d/test/"%str(num)
txt_path="./data/image/%d/txt/"%str(num)

#check="./data/check/"

path=train_path
txt_name='train.txt'


li=os.listdir(path)
li.sort()

jpgs=[]
for i in range(len(li)):
    if li[i][-3:]=='jpg':
        jpgs.append(li[i])
        
txt=[]
f=open(txt_path+txt_name, mode='w')
for i in range(len(jpgs)):
    img=cv2.imread(path+jpgs[i])
    col=img.shape[0]
    row=img.shape[1]
        
    roi_name=jpgs[i].split('.')[0]+'.roi'
    roi=read_roi_file(path+roi_name)

    name=jpgs[i].split('.')[0]
    
    try:
        x1_=roi[name]['left']
        y1_=roi[name]['top']
        x2_=roi[name]['left']+roi[name]['width']
        y2_=roi[name]['top']+roi[name]['height']
    except:
        x1_=roi[name]['x'][0]
        y1_=roi[name]['y'][0]
        x2_=roi[name]['x'][2]
        y2_=roi[name]['y'][2]

    x1=str(x1_)# round(x1_*512/row))
    y1=str(y1_)# round(y1_*512/col))
    x2=str(x2_)# round(x2_*512/row))
    y2=str(y2_)# round(y2_*512/col))

    if jpgs[i][0]=='N':
        clas='normal'
    if jpgs[i][0]=='A':
        clas='atypical'
    if jpgs[i][0]=='P':
        clas='positive'

    txt.append(path+jpgs[i]+','+x1+','+y1+','+x2+','+y2+','+clas)
    f.write(path+jpgs[i]+','+x1+','+y1+','+x2+','+y2+','+clas+'\n')
    print(txt[i])
f.close()    
'''
# draw box and check if it's right
for j in range(5):
    re=cv2.resize(img,(512,512))
    ch=cv2.rectangle(re,(round(x1_*512/row),round(y1_*512/col)),(round(x2_*512/row),round(y2_*512/col)),(255,0,0),3)
    cv2.imwrite(check+jpgs[i],re)
'''
    
#f=open(txt_path+'test.txt', mode='w')
#for i in range(len(txt)):
#    f.write(txt[i]+'\n')
#f.close()