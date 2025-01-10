import cv2, os, shutil
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from mask_utils import morphology
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="the compelx image path")
    return parser.parse_args()


def process(mask_path, sm_path, mask_out_path):
    mask = cv2.imread(mask_path, 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    mask = mask / 255

    sm = cv2.imread(sm_path, 0)
    _, sm = cv2.threshold(sm, 128, 255, cv2.THRESH_BINARY)
    sm = sm / 255

    res = cv2.bitwise_or(mask, sm)
    res = morphology(res, 5)
    cv2.imwrite(mask_out_path, np.uint8(res * 255))


if __name__ == '__main__':
    args = get_arguments()

    root = args.path
    image_path = os.path.join(root, "image")
    mask_path = os.path.join(root, "v1infer")
    sm_path = os.path.join(root, "v2infer")

    ouput_path = os.path.join(args.path, "v12infer")
    if not os.path.exists(ouput_path):
        os.makedirs(ouput_path)

    pool = Pool(processes=5)
    for jpg_name in tqdm(os.listdir(image_path)):
        png_name = jpg_name.replace("jpg", "png")

        mask_path_ = os.path.join(mask_path, png_name)
        sm_path_ = os.path.join(sm_path, png_name)

        mask_out_path = os.path.join(ouput_path, png_name)
        pool.apply_async(process, (mask_path_, sm_path_, mask_out_path))

    pool.close()
pool.join()








