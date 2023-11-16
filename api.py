import os
import sys
sys.path.append(os.path.dirname(__file__))

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
        if scores[j] <score_threshold:
            num_dets_to_consider = j
            break

    masks = masks[:num_dets_to_consider, :, :, None]
    class_names = []
    for j in range(num_dets_to_consider):
        class_names.append(cfg.dataset.class_names[classes[j]])

    return masks, boxes, class_names
    
    
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


def process_image(image: np.ndarray):
    global net

    # return the raw results
    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    masks, boxes, class_names = parse_result(preds, frame)

    # get the contour of each mask
    mask_contours = []
    for mask in masks:
        mask = mask.cpu().numpy()
        mask = np.uint8(mask) * 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # sort them based on the length
        contours = sorted(contours, key=lambda x: len(x[:, 0, :]), reverse=True)
        mask_contours.append(contours[0][:, 0, :].tolist())

    return masks, mask_contours, boxes, class_names
    

def get_recognition(image: np.ndarray):
    with torch.no_grad():
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        masks, mask_contours, boxes, class_names = process_image(image)

        return {
            'masks': masks,
            'mask_contours': mask_contours,
            'boxes': boxes,
            'class_names': class_names
        }