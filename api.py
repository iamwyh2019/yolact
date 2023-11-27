import os
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from yolact import Yolact
except ImportError:
    from .yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import cv2

import time

def parse_result(det_result, img, score_threshold = 0.15, top_k = 15):
    h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(det_result, w, h, visualize_lincomb = False,
                                        crop_masks        = False,
                                        score_threshold   = score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    if len(masks) == 0:
        masks = torch.reshape(masks, (0, h, w, 1))
    else:
        masks = masks[:num_dets_to_consider, :, :, None]

    class_names = []
    for j in range(num_dets_to_consider):
        class_names.append(cfg.dataset.class_names[classes[j]])

    # masks: torch.Tensor
    # boxes: np.ndarray
    # class_names: list
    # scores: np.ndarray
    return masks, boxes, class_names, scores
    
    
model_path = os.path.join(os.path.dirname(__file__), 'weights', 'yolact_base_54_800000.pth')
#'yolact_darknet53_54_800000.pth',
# 'yolact_resnet50_54_800000.pth',

net = Yolact()
net.load_weights(model_path)
net.eval()
net = net.cuda()

config_name = SavePath.from_str(model_path).model_name + '_config'
set_cfg(config_name)

net.detect.use_fast_nms = True
net.detect.use_cross_class_nms = False
cfg.mask_proto_debug = False


def process_image(image: np.ndarray, score_threshold = 0.15, top_k = 15):
    global net

    # return the raw results
    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    # masks: torch.Tensor, [N, H, W, 1]
    # boxes: np.ndarray, [N, 4]
    # class_names: list, [N]
    # scores: np.ndarray, [N]
    masks, boxes, class_names, scores = parse_result(preds, frame, score_threshold = score_threshold, top_k = top_k)

    masks = masks.cpu().numpy().astype(np.uint8)

    # get the contour of each mask
    geometry_centers = []
    mask_contours = []
    for mask in masks:
        mask_255 = mask * 255
        contours, hierarchy = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key = cv2.contourArea)

        mask_contours.append(largest_contour[:, 0, :].tolist()) # [N, 2]

        # get the geometry center of the mask
        M = cv2.moments(largest_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        geometry_centers.append([cx, cy])

    # remove the last dimension of masks
    masks = masks.squeeze(axis=3).tolist()

    # convert boxes to list
    boxes = boxes.tolist()

    # convert scores to list
    scores = scores.tolist()

    return masks, mask_contours, boxes, class_names, scores, geometry_centers
    

def get_recognition(image: np.ndarray, filter_objects = [], score_threshold = 0.15, top_k = 15):
    with torch.no_grad():
        cudnn.fastest = True 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        masks, mask_contours, boxes, class_names, scores, geometry_centers = process_image(image, score_threshold = score_threshold, top_k = top_k)

        if filter_objects:
            # filter the objects by name
            new_masks = []
            new_mask_contours = []
            new_boxes = []
            new_class_names = []
            new_scores = []
            new_geometry_centers = []
            for i, class_name in enumerate(class_names):
                if class_name in filter_objects:
                    new_masks.append(masks[i])
                    new_mask_contours.append(mask_contours[i])
                    new_boxes.append(boxes[i])
                    new_class_names.append(class_name)
                    new_scores.append(scores[i])
                    new_geometry_centers.append(geometry_centers[i])
            masks = new_masks
            mask_contours = new_mask_contours
            boxes = new_boxes
            class_names = new_class_names
            scores = new_scores
            geometry_centers = new_geometry_centers

        return {
            'masks': masks,
            'mask_contours': mask_contours,
            'boxes': boxes,
            'class_names': class_names,
            'scores': scores,
            'geometry_centers': geometry_centers,
        }
    
def show_recognition(image: np.ndarray, filter_objects = [], score_threshold = 0.15, top_k = 15):
    with torch.no_grad():
        cudnn.fastest = True 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        masks, mask_contours, boxes, class_names, scores, geometry_centers = process_image(image, score_threshold = score_threshold, top_k = top_k)

    if filter_objects:
        # filter the objects by name
        new_masks = []
        new_mask_contours = []
        new_boxes = []
        new_class_names = []
        new_scores = []
        new_geometry_centers = []
        for i, class_name in enumerate(class_names):
            if class_name in filter_objects:
                new_masks.append(masks[i])
                new_mask_contours.append(mask_contours[i])
                new_boxes.append(boxes[i])
                new_class_names.append(class_name)
                new_scores.append(scores[i])
                new_geometry_centers.append(geometry_centers[i])
        masks = new_masks
        mask_contours = new_mask_contours
        boxes = new_boxes
        class_names = new_class_names
        scores = new_scores
        geometry_centers = new_geometry_centers
   
    # draw the masks
    # each mask is a H*W 0/1 matrix, so multiply by a color to get the color mask
    for i, mask in enumerate(masks):
        mask = np.array(mask, dtype=np.uint8)
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask == 1] = [255, 0, 0]
        image = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

    # draw the contours
    for i, contour in enumerate(mask_contours):
        contour = np.array(contour, dtype=np.int32)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)
    
    # show the image
    cv2.imshow('image', image)
    cv2.waitKey(1)

if __name__ == '__main__':
    image = cv2.imread('./p1.png')
    show_recognition(image)