import numpy as np
import argparse
import re, os, cv2, torch, shutil
from segment_anything import SamAutomaticMaskGenerator,build_sam, SamPredictor,sam_model_registry

def get_arguments():
    parser = argparse.ArgumentParser()

    # normal settings
    parser.add_argument('--gpu', type=str, default='cuda:3')
    parser.add_argument('--sam_weight', type=str, default='/data/checkpoint/sam_vit_h_4b8939.pth', help="the sam model weight path")
    parser.add_argument('--th_salEdge', type=float, default=0.3, help="the threshold parameter of count_edge_pixels()")
    parser.add_argument('--morphology_kernal', type=int, default=5, help="the threshold parameter of morphology()")
    parser.add_argument('--th_miniSm', type=float, default=0.001, help="the parameter of min area sm")
    
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    
    # complex image settings
    parser.add_argument('--cinfer_target', type=str, default='infer')
    
    parser.add_argument('--cth_areaMask', type=float, default=0.9, help="the parameter of area proportion mask(0.9)")
    parser.add_argument('--cth_areaBox', type=float, default=0.9, help="the parameter of area proportion box(0.9)")
    parser.add_argument('--cth_maskIou', type=float, default=0.9, help="the parameter of mask iou(0.9)")

    # simple image settings
    parser.add_argument('--sth_nms', type=float, default=0.8, help="the parameter of mutil mask nms")
    parser.add_argument('--sth_clsProb1', type=float, default=0, help="the parameter of class probability")
    parser.add_argument('--sth_clsProb2', type=float, default=0.7, help="the parameter of strict class probability")
    parser.add_argument('--sth_maskIou', type=float, default=0.9, help="the parameter of mask iou")

    return parser.parse_args()

def get_sam_model(gpu, weight_path):
    sam = sam_model_registry["vit_h"](checkpoint=weight_path)
    sam.to(gpu)
    mask_generator = SamAutomaticMaskGenerator(model=sam, pred_iou_thresh=0.5, stability_score_thresh=0.8)
    sam_predictor = SamPredictor(build_sam(checkpoint=weight_path).to(gpu))
    sam.eval()
    return mask_generator, sam_predictor


# 计算显著性边界
def count_edge_pixels(mask, threshold, kernel_size=20):
    rows, cols = mask.shape
    # 膨胀像素
    count = 0
    kernel = np.ones((kernel_size,), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel)
  
    # 计算四周显著性区域
    count += np.sum(mask[0, :])
    count += np.sum(mask[rows-1, :])
    count += np.sum(mask[1:rows-1, 0])
    count += np.sum(mask[1:rows-1, cols-1])

    count /= 2*(rows + cols)
    return False if count > threshold else True

# 形态学操作 先腐蚀-膨胀，后膨胀腐蚀
def morphology(binary, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    res_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    res_close = cv2.morphologyEx(res_open, cv2.MORPH_CLOSE, kernel)
    return res_close

#根据初始掩码得到对应的Box
def get_propmt_by_mask(mask, min_area = 300, off = 20):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 符合条件的下标
    nice_indexs = [] 
    W, H = mask.shape[0], mask.shape[1]
    left, top = W, H
    right = bottom = -1
    
    # 挑选比较大的区域
    for index in range(len(contours)):
        area = cv2.contourArea(contours[index])
        if area > min_area: nice_indexs.append(index)

    # 获取边界框
    for item in nice_indexs:
        contour = contours[item]
        x, y, w, h = cv2.boundingRect(contour)
        left = min(left, x)
        top = min(top, y)
        right = max(right, x + w)
        bottom = max(bottom, y + h)

    # 对边界框进行缩放
    left = left - off if left > off else 0
    top = top - off if top > off else 0
    right = right + off if W - right > off else W
    bottom = bottom + off if H - bottom > off else H
    return [[left, top, right, bottom]]


# SAM-Box-Mode
def get_mask_by_sambox(config, sam_predictor, image, boxes):
    input_boxes = torch.tensor(boxes).to(config.gpu)
    sam_predictor.set_image(image)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    try:
        with torch.no_grad():
            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    except RuntimeError:
        return np.zeros(image.shape, dtype=np.uint8), False
    
    masks = ((masks.sum(dim=0)>0)[0]*1).cpu().numpy().astype('uint8')
    if len(masks.shape) == 3 and masks.shape[2] == 3:
        masks = cv2.cvtColor(masks, cv2.COLOR_BGR2GRAY)
    if count_edge_pixels(masks, config.th_salEdge):
        return masks, True
    else:
        return np.zeros(image.shape, dtype=np.uint8), False
