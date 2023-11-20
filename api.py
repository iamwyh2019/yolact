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


def process_image(image: np.ndarray):
    global net

    # return the raw results
    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    # masks: torch.Tensor, [N, H, W, 1]
    # boxes: np.ndarray, [N, 4]
    # class_names: list, [N]
    masks, boxes, class_names, scores = parse_result(preds, frame)

    # find geometric center of each mask using torch
    geometry_centers = []
    for i in range(masks.shape[0]):
        mask = masks[i, :, :, 0]
        indices = torch.nonzero(mask, as_tuple=True)
        y_indices, x_indices = indices
        y_center = torch.mean(y_indices.float()).item()
        x_center = torch.mean(x_indices.float()).item()
        geometry_centers.append([x_center, y_center])

    masks = masks.cpu().numpy().astype(np.uint8)

    # get the contour of each mask
    mask_contours = []
    for mask in masks:
        mask_255 = mask * 255
        contours, hierarchy = cv2.findContours(mask_255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # sort them based on the length
        contours = sorted(contours, key=lambda x: len(x[:, 0, :]), reverse=True)
        mask_contours.append(contours[0][:, 0, :].tolist())

    # remove the last dimension of masks
    masks = masks[:, :, :, 0]

    return masks, mask_contours, boxes, class_names, scores, geometry_centers
    

def get_recognition(image: np.ndarray):
    with torch.no_grad():
        cudnn.fastest = True 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        masks, mask_contours, boxes, class_names, scores, geometry_centers = process_image(image)

        
        # convert masks, scores and boxes to list
        masks = masks.tolist()
        boxes = boxes.tolist()
        scores = scores.tolist()

        return {
            'masks': masks,
            'mask_contours': mask_contours,
            'boxes': boxes,
            'class_names': class_names,
            'scores': scores,
            'geometry_centers': geometry_centers,
        }
    
def show_recognition(image: np.ndarray):
    with torch.no_grad():
        cudnn.fastest = True 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        masks, mask_contours, boxes, class_names, scores, geometry_centers = process_image(image)

    # draw the masks
    # each mask is a H*W 0/1 matrix, so multiply by a color to get the color mask
    for i, mask in enumerate(masks):
        # skip the first mask
        if i == 0:
            continue
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask == 1] = [255, 0, 0]
        image = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

    # draw the geometry centers and put the class name & scores above the center
    for i, center in enumerate(geometry_centers):
        cv2.circle(image, (int(center[0]), int(center[1])), 2, (0, 0, 255), -1)
        cv2.putText(image, class_names[i] + " " + str(scores[i]), (int(center[0]), int(center[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # show the image
    cv2.imshow('image', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    image = cv2.imread('./p1.png')
    show_recognition(image)