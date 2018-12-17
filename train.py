from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=10, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3-kitti.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/kitti.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/darknet53.conv.74", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/kitti.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()

my_dataset = opt.data_config_path
print('use'+ my_dataset)

vis = Visualizer('yolo v3')

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# check_point_path = 'checkpoints/'
# weights_files = os.listdir(check_point_path)
# weights_files = weights_files.sort()
# print(weights_files)
# weights_path_latest = check_point_path + weights_files[-1]
# print(weights_path_latest)
# Initiate model
model = Darknet(opt.model_config_path)
#model.apply(weights_init_normal)
model.load_weights(opt.weights_path)
#model.apply(weights_init_normal)

if cuda:
    model = model.cuda()
    print("CUDA is ready")

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
freeze_backbone = 1
print("start traing")

for epoch in range(opt.epochs):
    losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
    
        # Freeze darknet53.conv.74 layers for first epoch
    if freeze_backbone:
        if epoch == 0:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < 75:  # if layer < 75
                    p.requires_grad = False
        elif epoch == 1:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < 75:  # if layer < 75
                    p.requires_grad = True
                    
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()
        
        losses_x += model.losses["x"]
        losses_y += model.losses["y"]
        losses_w += model.losses["w"]
        losses_h += model.losses["h"]
        losses_conf += model.losses["conf"]
        losses_cls += model.losses["cls"]
        losses_recall += model.losses["recall"]
        losses_precision += model.losses["precision"]
    
        batch_loss += loss.item()
        
        if (batch_i+1) % 50 == 0:
            vis.plot('losses_x',losses_x)
            vis.plot('losses_y',losses_y)
            vis.plot('losses_w',losses_w)
            vis.plot('losses_h',losses_h)
            vis.plot('losses_conf',losses_conf)
            vis.plot('losses_cls',losses_cls)
            vis.plot('losses_recall',losses_recall)
            vis.plot('losses_precision',losses_precision)
            vis.plot('batch_loss',batch_loss)
            losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
            
        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
