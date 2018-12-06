import numpy as np
import cv2
import os
import sys

coco_img_path = 'coco/images/'
coco_label_path = 'coco/raw_labels/'
coco_trans_label_path = 'coco/trans_labels/'
index = 0
cvfont = cv2.FONT_HERSHEY_SIMPLEX

coco_names = open('coco.names','r')
coco_names_contents = coco_names.readlines()                
coco_images = os.listdir(coco_img_path)
coco_labels = os.listdir(coco_label_path)
coco_images.sort()
coco_labels.sort()
#创建训练集图片的List
f = open('trainvalno5k-coco.txt','w')
for img in coco_images:
    f.write(coco_img_path+img+'\n')
f.close()

#coco数据集 相对坐标 转换为绝对坐标
for indexi in range(len(coco_images)):
    j = '*'
    coco_img_totest_path = coco_img_path + coco_images[indexi]
    coco_label_totest_path = coco_label_path + coco_labels[indexi]
    coco_img_totest = cv2.imread(coco_img_totest_path)
    img_height, img_width = coco_img_totest.shape[0],coco_img_totest.shape[1]
    coco_label_totest = open(coco_label_totest_path,'r')
    label_contents = coco_label_totest.readlines()
    real_label = open(coco_trans_label_path+coco_labels[indexi],'w')
    for line in label_contents:
        data = line.split(' ')
        x=y=w=h=0
        if(len(data) >= 5):
            class_num = int(data[0])
            #(x,y) center (w,h) size
            x = float(data[1])
            y = float(data[2])
            w = float(data[3])
            h = float(data[4])
    #         print(class_num,x,y,w,h)
    #         real_x = round(x*img_width) 
    #         real_y = round(y*img_height)
    #         r_w = round(w*img_width)
    #         r_h = round(h*img_height)
            x1 = round(img_width*(x - w/2))
            y1 = round(img_height*(y - h/2))
            x2 = round(img_width*(x  + w/2))
            y2 = round(img_height*(y + h/2))
            #print(coco_names_contents[class_num])
            # cv2.putText()
            # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
            cv2.putText(coco_img_totest, coco_names_contents[class_num].rstrip(), (x1, y1+3), cvfont, 2, (0,0,255), 1)
            # cv2.rectangle()
            # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
            cv2.rectangle(coco_img_totest, (x1,y1), (x2,y2), (0,255,0), 2)
            line_to_write = coco_names_contents[class_num].rstrip() + ' ' + str(x1)+ ' ' + str(y1)+ ' ' + str(x2)+ ' ' + str(y2) +'\n'
            real_label.write(line_to_write)
            j+='*'
            sys.stdout.write(str(int((indexi/len(coco_images))*100))+'%  ||'+j+'->'+"\r")
            sys.stdout.flush()
    #window_name = 
    cv2.imshow(str(indexi)+' coco_label_show',coco_img_totest)    
    cv2.waitKey()
    real_label.close()
coco_names.close()