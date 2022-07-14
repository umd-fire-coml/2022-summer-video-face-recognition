# Import required libraries
import argparse
import pwd
import random
from pathlib import Path
import util.misc as utils

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import PIL.Image
from checkpoints import trainedweights
from models import build_model
from main import get_args_parser


trainedweights()
parser = argparse.ArgumentParser(description="DETR Args Parser", parents=[get_args_parser()])
args = parser.parse_args(args=[])
args.resume = 'trained.pth'
args.device = 'cpu'

if args.output_dir:
  Path(args.output_dir).mkdir(parents=True, exist_ok=True)

args.distributed = False
model, criterion, postprocessors = build_model(args)

device = torch.device(args.device)
model.to(device)
output_dir = Path(args.output_dir)
##new block

output_dir = Path(args.output_dir)
if args.resume:
  
  checkpoint = torch.load(args.resume, map_location='cpu')
  model.load_state_dict(checkpoint['model'], strict=True)
model.load_state_dict(checkpoint['model'], strict=True)
##new block
COLORS = [[1.000,1.000,1.000]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
CLASSES = [
   'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
   'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
   'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
   'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
   'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
   'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
   'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
   'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
   'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
   'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
   'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
   'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
   'toothbrush'
]
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# Rescale bounding boxes to be full size of the image
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform):
  # Mean-STD Normalize the input image (of batch-size 1)
  img = transform(im).unsqueeze(0)
  # The demo model only supports images with aspect ratios between 0.5 and 2.0
  # If you want to use images with an aspect ratio outside this range,
  # Resize the image so the maximum size is at most 1333 for best results
  assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'Demo model only supports images up to 1600 on each side.'
  # Propogate through the model on the input image
  outputs = model(img)
  # Keep only the predictions with 0.7+ confidence
  probas = outputs['pred_logits'].softmax(-1)[0,:,:-1]
  keep = probas.max(-1).values > 0.7
  # Convert bboxes from [0;1] scale to image scale
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  # Return probabilities and scaled bboxes
  return probas[keep], bboxes_scaled


def plot_results(pil_img, class_probs, b_boxes, show, classes, is_ground_truth=False):
    plt.figure(figsize=(12,6))
    plt.axis('on')
    ax = plt.gca()
    ax.imshow(pil_img)
    measurements = []
    for p, (xmin, ymin, xmax, ymax), c in zip(class_probs, b_boxes, COLORS * 100):
        cl = p if is_ground_truth else p.argmax()
        
        # If the class isn't present, skip this annotation
        if CLASSES[cl] not in classes:
            continue
        # Plot bounding box and label (Note difference in bounding box format, xmax vs xmax-xmin)
        xmax = xmax if is_ground_truth else xmax-xmin
        ymax = ymax if is_ground_truth else ymax-ymin
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax, ymax, fill=False, color=c, linewidth=3))
        measurements.append(plt.Rectangle((xmin, ymin), xmax, ymax))
    # Show the plot
    if(show):
      plt.show()

    return [[x.get_xy(), x.get_width(), x.get_height()] for x in measurements]