# Kitti label trans to COCO label 

## this this a tool to tranfrom these two datasets labels.
use`kitti2coco-label-trans.py' to tansfrom labels.

before you run this scirpt, you should set your data sets absolute path:
`kitti_img_path` and `kitti_label_path`

and `kitti_label_tosave_path` will save these coordinate transformed label files.
and `train.txt` is your kitti train set all images' absolute path, put it into `data/kitti` folder

## Test on ubuntu 18.04 with python 3.6
