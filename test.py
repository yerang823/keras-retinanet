import os

os.system('python keras_retinanet/bin/train.py --steps=1000 --workers=0 --snapshot-path ./snapshots0 csv data/image/0/txt/train.csv data/1fold/txt/class_name.csv')

os.system('python keras_retinanet/bin/train.py --steps=1000 --workers=0 --snapshot-path ./snapshots1 csv data/image/1/txt/train.csv data/1fold/txt/class_name.csv')

os.system('python keras_retinanet/bin/train.py --steps=1000 --workers=0 --snapshot-path ./snapshots2 csv data/image/2/txt/train.csv data/1fold/txt/class_name.csv')

os.system('python keras_retinanet/bin/train.py --steps=1000 --workers=0 --snapshot-path ./snapshots3 csv data/image/3/txt/train.csv data/1fold/txt/class_name.csv')

os.system('python keras_retinanet/bin/train.py --steps=1000 --workers=0 --snapshot-path ./snapshots4 csv data/image/4/txt/train.csv data/1fold/txt/class_name.csv')