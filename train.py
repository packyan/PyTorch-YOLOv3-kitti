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
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=10, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3-kitti.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/kitti.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="checkpoints/darknet53.conv.74", help="path to weights file/ can use darknet53.conv.74")
parser.add_argument("--class_path", type=str, default="data/kitti.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)
my_dataset = opt.data_config_path
print('use'+ my_dataset)
freeze_backbone = 1
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
test_path = data_config["valid"]
num_classes = int(data_config["classes"])

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
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
accumulated_batches = 4
best_mAP = 0.0


print("start traing")
loss_data_file = open('loss data.txt','w+')
test_data_file = open('test_data.txt','w+')

#use for test, get AP on valid test
test_dataset = ListDataset(test_path)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=opt.n_cpu) 


for epoch in range(opt.epochs):
    losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
    # Freeze darknet53.conv.74 layers for first some epochs
    if freeze_backbone:
        if epoch < 20:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < 75:  # if layer < 75
                    p.requires_grad = False
        elif epoch >= 20:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < 75:  # if layer < 75
                    p.requires_grad = True
                    
    optimizer.zero_grad()   
                   
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
       
        loss = model(imgs, targets)

        loss.backward()
        #optimizer.step()
                # accumulate gradient for x batches before optimizing
        if ((batch_i + 1) % accumulated_batches == 0) or (batch_i == len(dataloader) - 1):
            optimizer.step()
            optimizer.zero_grad()
            
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
            loss_data_file.flush()
            vis.plot('Losses:x',losses_x)
            vis.plot('Losses:y',losses_y)
            vis.plot('Losses:w',losses_w)
            vis.plot('Losses:h',losses_h)
            vis.plot('Losses:conf',losses_conf)
            vis.plot('Losses:cls',losses_cls)
            vis.plot('Recall',losses_recall)
            vis.plot('Precision',losses_precision)
            vis.plot('Total Loss',batch_loss)
            losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
        
        #imgs_num = 1
        loss_data = "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n"% (
                model.losses["x"] ,
                model.losses["y"] ,
                model.losses["w"] ,
                model.losses["h"] ,
                model.losses["conf"] ,
                model.losses["cls"] ,
                loss.item(),
                model.losses["recall"] ,
                model.losses["precision"],
            )
        loss_data_file.write(loss_data)
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
        
    #mean_AP = test_model(test_dataloader, test_data_file, model)
    print("Compute %d Epoch mAP..." % epoch)

    all_detections = []
    all_annotations = []

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(test_dataloader, desc="Detecting objects")):

        imgs = Variable(imgs.type(Tensor))

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, 80, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

        for output, annotations in zip(outputs, targets):

            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                # Get predicted boxes, confidence scores and labels
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                # Order by confidence
                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotations[:, -1] > 0):

                annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= opt.img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    print("Average Precisions:")
    for c, ap in average_precisions.items():
        print(f"+ Class '{c}' - AP: {ap}")
        test_data_file.write("%.5f "%ap)
    mAP = np.mean(list(average_precisions.values()))
    print(f"mAP: {mAP}")
    test_data_file.write("%.5f\n"% mAP)
    
    if(mAP > best_mAP):
        best_mAP = mAP
        model.save_weights("%s/kitti_best.weights" % (opt.checkpoint_dir))
        print("New Best AP appear !!! %f" % best_mAP)
        test_data_file.flush()
    
loss_data_file.close()
test_data_file.close()
