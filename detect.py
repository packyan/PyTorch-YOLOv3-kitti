from __future__ import division

from pathlib import Path

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

kitti_weights = 'weights/yolov3-kitti.weights'
save_dir = 'output/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples/', help='path to dataset')  # 不知道为啥，前面会多一个“/”
parser.add_argument('--config_path', type=str, default='config/yolov3-kitti.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default=kitti_weights, help='path to weights file')
parser.add_argument('--class_path', type=str, default='label_transform/kitti.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=2, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=0,
                    help='number of cpu threads to use during batch generation')  # windows上num_workers为0
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print('Config:')
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)
print('model path: ' + opt.weights_path)
if cuda:
    model.cuda()
    print("using cuda model")

model.eval()  # Set in evaluation mode
# image_folder = opt.image_folder[1:]  # 不知道为啥，前面会多一个“/”
image_folder = opt.image_folder
dataset = ImageFolder(image_folder, img_size=opt.img_size)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

print('data size : %d' % len(dataloader))
print('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        # print(detections)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
        # print(detections)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
# cmap = plt.get_cmap('tab20b')
# cmap = plt.get_cmap('Vega20b')
cmap = plt.get_cmap('Blues')  # 估计是新版本，里面没有Vega20b。根据提示，选了一个Blues
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print('\nSaving images:')
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # kitti_img_size = 11*32
    kitti_img_size = 416
    # The amount of padding that was added
    # pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    # pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = kitti_img_size - pad_y
    unpad_w = kitti_img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        print(type(detections))
        print(detections.size())
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
            # Rescale coordinates to original dimensions
            box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
            box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
            y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
            x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1 - 30, s=classes[int(cls_pred)] + ' ' + str('%.4f' % cls_conf.item()), color='white',
                     verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())

    img_path = '{}{}.png'.format(save_dir, img_i)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()
