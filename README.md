# DOTA database training with yolo | 基于DOTA数据集的yolo训练

![result example](./output.jpg)

## clone this repo | 克隆此仓库

```
git clone xxxx/dota-yolo
cd dota-yolo
```

## download DOTA dataset | 下载数据集

download from | 这里下载 https://captain-whu.github.io/DOTA/dataset.html

then put them as following structure | 然后按如下结构存放

```
DOTA-yolov3
|
├─dataset
│  ├─train
│  │  ├─images
│  │  └─labelTxt
│  └─val
│      ├─images
│      └─labelTxt
```

## prerequisites | 必要环境

- darknent https://pjreddie.com/darknet/install/
- python3
- opencv-python etc. | opencv-python 等必要包
- Shapely (need geos|需要安装geos)

## split | 切割

```
python3 data_transform/split.py
```

for more details refer https://github.com/CAPTAIN-WHU/DOTA_devkit

## transform label | 转换label

```
mkdir dataset/trainsplit/labels
mkdir dataset/valsplit/labels
python3 data_transform/YOLO_Transform.py

# check labels
# cd dataset/trainsplit/labels
# awk -F" " '{col[$1]++} END {for (i in col) print i, col[i]}' *.txt

# 

```

for more details refer https://github.com/ringringyi/DOTA_YOLOv2

## generate | 生成 train.txt & val.txt

```
ls -1d $PWD/dataset/trainsplit/images/* > cfg/train.txt
ls -1d $PWD/dataset/valsplit/images/* > cfg/val.txt

# when too many images
# find "$(pwd)/dataset/trainsplit512/images">cfg/train.txt
# find "$(pwd)/dataset/valsplit512/images">cfg/val.txt
```

## train | 训练

```
cd cfg
mkdir backup

# yolo-tiny
darknet detector train dota.data dota-yolov3-tiny.cfg 

# more gpus
darknet detector train dota.data dota-yolov3-tiny.cfg -gpus 0,1,2

# resume from unexpected stop
darknet detector train dota.data dota-yolov3-tiny.cfg backup/dota-yolov3-tiny.backup

# or yolov3-416
darknet detector train dota.data dota-yolov3-416.cfg 
```

config cfg files refer https://medium.com/@manivannan_data/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2

## predict | 预测

为了方便运行，这里用 opnencv做检测，要高性能还是用 darknent binding | Here I prepared opencv for easy set up, you can use darknet bindings for better performance

```
# tiny
python test.py --image test.png --config cfg/dota-yolov3-tiny.cfg --weights cfg/backup/dota-yolov3-tiny_final.weights --classes cfg/dota.names

# or 416
python test.py --image test.png --config cfg/dota-yolov3-416.cfg --weights cfg/backup/dota-yolov3-416_final.weights --classes cfg/dota.names

```

## pretraind params | 预训练的参数

百度云 链接: https://pan.baidu.com/s/1V6fxrUpHLhukiJ-vT0F0pQ 提取码: pfq6

google drive: https://drive.google.com/drive/folders/1Y-W2npeaqflfO8IUA7gx9PzmesaSl9rY?usp=sharing
