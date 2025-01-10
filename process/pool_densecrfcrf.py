import os
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from PIL import Image
from  multiprocessing import Pool
from pydensecrf.utils import unary_from_labels

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

def normalize_pil(pred):
    pred = np.asarray(pred)
    pred = cv2.threshold(pred, 0, 255, cv2.THRESH_OTSU)[1]
    max_pre, min_pre = pred.max(), pred.min()
    pred = pred / 255  if max_pre == min_pre else (pred - min_pre) / (max_pre - min_pre)
    return pred

def crf_inference_label(img, labels, t=10, n_labels=2, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)
    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def process(index, _img_path, _mask_path, _out_path):
    image = Image.open(_img_path).convert('RGB')
    image = np.array(image).astype(np.uint8)
    mask = Image.open(_mask_path).convert('L')

    mask = normalize_pil(mask)
    mask = (mask > 0.5).astype(np.uint8)
    mask = crf_inference_label(image, mask)
    mask = np.clip(np.round(mask * 255) / 255., 0, 1) * 255
    mask = cv2.medianBlur(np.uint8(mask), 9)
    cv2.imwrite(_out_path, mask)
    print(index, "done!", end="\r")

def main(img_path, mask_path, output_path):
    pool = Pool(8)
    length = len(os.listdir(img_path))
    for index, img_name in enumerate(os.listdir(img_path)):
        png_name = img_name.replace("jpg","png")
        _img_path = os.path.join(img_path, img_name)
        _mask_path = os.path.join(mask_path,png_name)
        _out_path = os.path.join(output_path, png_name)
        idx = f"{index}/{length}"
        # process(idx, _img_path, _mask_path, _out_path)
        pool.apply_async(process,(idx, _img_path, _mask_path, _out_path))
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    img_path = "/opt/FOUND/data/DUTS-TR/image"
    mask_path = "/opt/FOUND/outputs/mask-aux-solver"
    output_path = "/opt/ObjectDiscovery/dataset/DUTS-TR/mask-aux"
    if not os.path.exists(output_path): os.makedirs(output_path)
    main(img_path, mask_path, output_path)






    

