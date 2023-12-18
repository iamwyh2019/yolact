import os
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from yolact import Yolact
except ImportError:
    from .yolact import Yolact
try:
    from data import COLORS, cfg, set_cfg
except ImportError:
    from .data import COLORS, cfg, set_cfg

from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation

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
    
    
# model_path = os.path.join(os.path.dirname(__file__), 'weights', 'yolact_base_54_800000.pth')
model_path = os.path.join(os.path.dirname(__file__), 'weights', 'yolact_darknet53_54_800000.pth')
# 'yolact_resnet50_54_800000.pth',

config_name = SavePath.from_str(model_path).model_name + '_config'
set_cfg(config_name)

cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = Yolact()
net.load_weights(model_path)
net.eval()
net = net.cuda()

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
    # mask is not accurate, but bounding box is
    # so we first filter out the areas outside the bounding box
    # then find the contour and get the largest one
    mask_contours = []
    geometry_center = []
    for i, mask in enumerate(masks):
        x1, y1, x2, y2 = boxes[i]
        mask_crop = mask[y1:y2, x1:x2]

        # mask_crop is 0/1 matrix
        # the center is the center of mass (assume uniform density)
        # the contour is the largest contour
        center = cv2.moments(mask_crop)
        center_x = int(center["m10"] / center["m00"]) + x1
        center_y = int(center["m01"] / center["m00"]) + y1
        geometry_center.append([center_x.item(), center_y.item()])

        contours, hierarchy = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key = cv2.contourArea)
        largest_contour = largest_contour[:, 0, :] + np.array([x1, y1]) # [N, 2]
        mask_contours.append(largest_contour.tolist())

    return masks, mask_contours, boxes, class_names, scores, geometry_center

def get_filtered_objects(mask, mask_contours, boxes, class_names, scores, geometry_center, filter_objects = []):
    new_masks = []
    new_mask_contours = []
    new_boxes = []
    new_class_names = []
    new_scores = []
    new_geometry_center = []
    for i, class_name in enumerate(class_names):
        if class_name in filter_objects:
            new_masks.append(mask[i])
            new_mask_contours.append(mask_contours[i])
            new_boxes.append(boxes[i])
            new_class_names.append(class_name)
            new_scores.append(scores[i])
            new_geometry_center.append(geometry_center[i])
    return new_masks, new_mask_contours, new_boxes, new_class_names, new_scores, new_geometry_center
    

def get_recognition(image: np.ndarray, filter_objects = [], score_threshold = 0.15, top_k = 15):
    with torch.no_grad():
        masks, mask_contours, boxes, class_names, scores, geometry_center = process_image(image, score_threshold = score_threshold, top_k = top_k)

    masks = masks.squeeze(3).tolist()
    boxes = boxes.tolist()
    scores = scores.tolist()

    # scores to 2 decimal places
    scores = [round(score, 2) for score in scores]

    if filter_objects:
        # filter the objects by name
        masks, mask_contours, boxes, class_names, scores, geometry_center = get_filtered_objects(masks, mask_contours, boxes, class_names, scores, geometry_center, filter_objects = filter_objects)

    return {
        'masks': masks,
        'mask_contours': mask_contours,
        'boxes': boxes,
        'class_names': class_names,
        'scores': scores,
        'geometry_center': geometry_center,
    }

def draw_recognition(image: np.ndarray, result,
                     black = False, draw_contour = False, draw_mask = True, 
                     draw_box = False, draw_text = True, alpha = 0.45):
    masks = result['masks']
    mask_contours = result['mask_contours']
    boxes = result['boxes']
    class_names = result['class_names']
    scores = result['scores']
    geometry_center = result['geometry_center']
    
    if black:
        image = np.zeros_like(image)
    
    if draw_mask:
    # draw the masks
    # each mask is a H*W 0/1 matrix, so multiply by a color to get the color mask
        for i, mask in enumerate(masks):
            color = COLORS[i*5 % len(COLORS)]
            color = (color[2], color[1], color[0])
            color_mask = mask * alpha * np.array(color, dtype=np.uint8)

            image = ((image * (1-mask)) + (image * mask * (1-alpha) + color_mask)).astype(np.uint8)

    # draw the contours
    if draw_contour:
        for i, contour in enumerate(mask_contours):
            contour = np.array(contour, dtype=np.int32)
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # draw box
    if draw_box:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # place text at the center
    if draw_text:
        for i, center in enumerate(geometry_center):
            text = class_names[i] + ' ' + str(round(scores[i], 2))
            cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return image
    
def show_recognition(image: np.ndarray, filter_objects = [], score_threshold = 0.15, top_k = 15, alpha = 0.45,
                     black = False, draw_contour = False, draw_text = True,
                     draw_mask = True, draw_box = False):
    result = get_recognition(image, filter_objects = filter_objects, score_threshold = score_threshold, top_k = top_k)

    image = draw_recognition(image, result,
                            black = black, draw_contour = draw_contour,
                            draw_mask = draw_mask, draw_box = draw_box,
                            draw_text = draw_text, alpha = alpha)
    return image

if __name__ == '__main__':
    image = cv2.imread('./p1.png')
    show_recognition(image)