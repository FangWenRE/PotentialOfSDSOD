import numpy as np
import time
import re, os, cv2, torch, shutil
from mask_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed


# SAM-Everthing-Mode
def process_everthing_mask(config, mask_generator, image_rgb, sm, box_mask):
    with torch.no_grad():
        sam_result = mask_generator.generate(image_rgb)
    sal_map = np.zeros(sm.shape, dtype=np.uint8)
    flag = False
    for _, mask in enumerate(sam_result):
        mask = np.array(mask["segmentation"], dtype=np.uint8)
        if not count_edge_pixels(mask, config.th_salEdge): continue

        area_proportion_mask = np.sum(cv2.bitwise_and(mask, sm)) / np.sum(mask)
        area_proportion_box = np.sum(cv2.bitwise_and(mask, box_mask)) / np.sum(mask)

        # print(f" * {index} {area_proportion_mask:.4f} {area_proportion_box:.4f}")
        if area_proportion_mask > config.cth_areaMask or area_proportion_box > config.cth_areaBox:
            flag = True
            sal_map += mask
    return sal_map, flag


# Main processing function
# Jointly use box and salient map to determine whether the segmentation result of everything is valid
def process(config, models, image_path, infer_path, output_image_path, output_mask_path):
    mask_generator, sam_predictor = models

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sm = cv2.imread(infer_path, 0)
    sm = cv2.threshold(sm, 64, 255, cv2.THRESH_BINARY)[1]

    if np.count_nonzero(sm) / sm.size < config.th_miniSm: return

    boxes = get_propmt_by_mask(sm)
    sal_map_box, flag_box = get_mask_by_sambox(config, sam_predictor, image_rgb, boxes)
    if not flag_box: return

    sm = np.uint8(sm / sm.max())
    sal_map_ever, flag_ever = process_everthing_mask(config, mask_generator, image_rgb, sm, sal_map_box)
    sal_map_ever[sal_map_ever > 0] = 1

    if flag_ever:
        mask_iou = np.sum(cv2.bitwise_and(sal_map_ever, sal_map_box)) / (
                    np.sum(cv2.bitwise_or(sal_map_ever, sal_map_box)) + 1e-8)
        if mask_iou > config.cth_maskIou:
            print(f"*pielx IoU: {mask_iou:.4f}")
            sal_map_ever = morphology(sal_map_ever, config.morphology_kernal)
            cv2.imwrite(output_mask_path, np.uint8(sal_map_ever * 255))
            shutil.copy(image_path, output_image_path)


def main(config):
    mask_generator, sam_predictor = get_sam_model(config.gpu, config.sam_weight)
    models = mask_generator, sam_predictor

    # Original image directory
    root = config.input_path
    images = os.listdir(os.path.join(root, "image"))

    # Output image and mask directory.
    out_root = config.output_path
    out_image_path = os.path.join(out_root, "image/")
    out_mask_path = os.path.join(out_root, "mask/")
    if not os.path.exists(out_image_path): os.makedirs(out_image_path)
    if not os.path.exists(out_mask_path): os.makedirs(out_mask_path)

    tar_len = len(images)
    for index, image_name in enumerate(images, start=1):
        print(f"{index}/{tar_len}, {image_name}")
        png_name = image_name.replace(".jpg", ".png")

        image_path = os.path.join(root, "image", image_name)
        infer_path = os.path.join(root, config.cinfer_target, png_name)
        out_mask_path_ = os.path.join(out_mask_path, png_name)
        out_image_path_ = os.path.join(out_image_path, image_name)
        if not os.path.exists(infer_path): continue

        process(config, models, image_path, infer_path, out_image_path_, out_mask_path_)


if __name__ == "__main__":
    config = get_arguments()
    print(config)
    main(config)
    print("Done!")

# nohup python -u process/generate_complex_mask.py --gpu="cuda:1" > ./logs/labels/1stage02_2.log 2>&1 &
