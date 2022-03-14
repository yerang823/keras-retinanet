# Result
![image](https://user-images.githubusercontent.com/97432613/158131314-1c5be973-9ec5-4cc8-bc30-5f86fe8b42de.png)


# Architecture
![image](https://user-images.githubusercontent.com/97432613/158131513-b24cb5f3-b916-4fe9-9ad8-494b123650cc.png)


# Usage 

### Preprocess
```python train_test_split_txt.py```
- When using k-fold cross validation, split annotation txt files.

### Train
```python keras_retinanet/bin/train.py --steps=1000 --workers=0 csv data/txt/train.csv data/txt/class_name.csv```

### Convert model
```python keras_retinanet/bin/convert_model.py ./snapshots0/resnet50_csv_50.h5 ./final_model/resnet50_final.h5```

### Predict
```python ResNet50RetinaNet.py```
- Save result of images and txt files.

### Evaluation
```python evaluate_classification.py```
