#num=4


for num in range(1):#5):
  # txt_path="./data/txt/test_label.txt"
  #label_path="./result/%s/txt/test_pred_201015.txt"%str(num)
  label_path="./result/210114_test/txt/test_pred_N1.txt"
  
  f=open(label_path,"r")
  labels=f.readlines()
  f.close()
  
  y_test=[]
  pred_label=[]
  for i in range(len(labels)):
      try:
          file=labels[i].split(',')[0]
          t_clas=file.split('/')[-1][0] # A,N,P
          p_clas=labels[i].split(',')[-1] # normal,positive
                  
          y_test.append(t_clas)
          pred_label.append(p_clas)
          
      except:
          print(labels[i])
  
  for i in range(len(pred_label)):
      if pred_label[i]=='normal\n' or pred_label[i]=='noraml\n':
          pred_label[i]=0
          
      elif pred_label[i]=='positive\n':
          pred_label[i]=1
      
      #elif pred_label[i]=='positive\n':
      #    pred_label[i]=2
          
  for i in range(len(y_test)):
      if (y_test[i]=='N' or y_test[i]=='A'):
          y_test[i]=0
          
      elif y_test[i]=='P':
          y_test[i]=1
      
      #elif y_test[i]=='P':
      #    y_test[i]=2
          
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import accuracy_score
  from keras.preprocessing.image import array_to_img
  from keras.models import Model, load_model
  
  print(num,"=======================")
  cm=confusion_matrix(y_test,pred_label)
  print(cm)
  acc=accuracy_score(y_test,pred_label)
  print("acc=",acc)
  
  print(cm[0][0]/(cm[0][0]+cm[0][1]))
  print(cm[1][1]/(cm[1][0]+cm[1][1]))
  #print(cm[2][2]/(cm[2][0]+cm[2][1]+cm[2][2]))