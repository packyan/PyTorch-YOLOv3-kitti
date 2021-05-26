# Kitti label trans to COCO label 

## this is a tool to tranfrom these two datasets labels.
use`kitti2coco-label-trans.py` to tansfrom labels.

before you run this scirpt, you should set your data sets absolute path:
`kitti_img_path` and `kitti_label_path`（此为kitti2coco-label-trans.py文件中的变量）

and `kitti_label_tosave_path` will save these coordinate transformed label files.
and `train.txt` is your kitti train set all images' absolute path, put it into `data/kitti` folder

###### Test on ubuntu 18.04 with python 3.6

————————————————————以上为原作者：——————————————————————————

————————————————————中文：——————————————————————————————

`kitti2coco-label-trans.py`将kitti转换成coco数据格式

文件中的变量：

`kitti_img_path` ：kitti数据集图片文件夹

`kitti_label_path`：kitti数据集label文件夹

`kitti_label_tosave_path` ：转化成coco数据集形式后，要存放在哪里

`train.txt` ：运行`kitti2coco-label-trans.py`后，生成的训练集包含哪些图片（图片地址）