# Kitti label trans to COCO label 

`kitti2coco-label-trans.py`将kitti转换成coco数据格式

#### py文件中的变量：

`kitti_img_path` ：kitti数据集图片文件夹

`kitti_label_path`：kitti数据集label文件夹

`kitti_label_tosave_path` ：转化成coco数据集形式后，要存放在哪里

`train.txt` ：运行`kitti2coco-label-trans.py`后，生成的训练集包含哪些图片（图片地址）

Test on ubuntu 18.04 with python 3.6



使用方法：

- 修改上述变量
- 将pycharm工作目录换成根目录