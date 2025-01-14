import numpy as np
import re, os, cv2, torch, open_clip, clip, shutil
from PIL import Image
from mask_utils import *
import time
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# HF_ENDPOINT=https://hf-mirror.com python XX.py

config = get_arguments()
device = config.gpu
mask_generator, sam_predictor = get_sam_model(device, config.sam_weight)

CLS_NAMES = ['house', 'centipede', 'tape player', 'kite', 'piano', 'crutch', 'desk', 'fox', 'suitcase', 'surfboard',
             'ruler', 'helmet', 'wine glass', 'french horn', 'fire hydrant', 'salt or pepper shaker', 'snowboard',
             'teddy bear', 'computer keyboard', 'dining table', 'swine', 'drum', 'ray', 'snail', 'porcupine',
             'backpack', 'microphone', 'bathing cap', 'pencil box', 'mouse', 'starfish', 'artichoke', 'toothbrush',
             'harmonica', 'remote', 'toaster', 'strawberry', 'nail', 'baseball', 'carrot', 'potted plant', 'couch',
             'punching bag', 'digital clock', 'motorcycle', 'car', 'computer mouse', 'trumpet', 'elephant',
             'rugby ball', 'cart', 'bee', 'swimming trunks', 'waffle iron', 'otter', 'plate rack', 'harp', 'golf ball',
             'frying pan', 'table', 'flute', 'seal', 'washer', 'bell pepper', 'ladle', 'bear', 'chair', 'bagel', 'oboe',
             'domestic cat', 'laptop', 'hamster', 'basketball', 'rabbit', 'ski', 'head cabbage', 'lizard', 'skateboard',
             'chime', 'cello', 'fork', 'coffee maker', 'knife', 'popsicle', 'refrigerator', 'corkscrew',
             'rubber eraser', 'tennis ball', 'bookshelf', 'baby bed', 'bicycle', 'goldfish', 'orange', 'pineapple',
             'squirrel', 'tennis racket', 'stove', 'hot dog', 'balance beam', 'binder', 'miniskirt', 'remote control',
             'horse', 'toilet', 'sunglasses', 'bow', 'beaker', 'umbrella', 'skis', 'lamp', 'brassiere', 'diaper',
             'mushroom', 'hat with a wide brim', 'pretzel', 'stretcher', 'window', 'flower pot', 'plastic bag', 'sofa',
             'perfume', 'handbag', 'lion', 'vacuum', 'red panda', 'traffic light', 'camel', 'face powder', 'whale',
             'unicycle', 'printer', 'cattle', 'hamburger', 'fig', 'frisbee', 'baseball bat', 'dumbbell', 'dishwasher',
             'donut', 'ping-pong ball', 'tv', 'croquet ball', 'neck brace', 'bus', 'bench', 'koala bear', 'iPod',
             'hair drier', 'filing cabinet', 'cell phone', 'dog', 'volleyball', 'train', 'scorpion', 'cup',
             'baseball glove', 'microwave', 'antelope', 'chain saw', 'watercraft', 'scissors', 'frog', 'stethoscope',
             'bird', 'hammer', 'person', 'cow', 'axe', 'giant panda', 'airplane', 'isopod', 'racket', 'band aid',
             'soccer ball', 'milk can', 'banjo', 'dragonfly', 'pitcher', 'puck', 'keyboard', 'pomegranate', 'hotdog',
             'banana', 'skunk', 'strainer', 'book', 'can opener', 'golfcart', 'pizza', 'truck', 'soap dispenser',
             'clock', 'stop sign', 'apple', 'bowl', 'hair dryer', 'giraffe', 'door', 'lipstick', 'broccoli', 'tie',
             'blender', 'maraca', 'cream', 'maillot', 'armadillo', 'hippopotamus', 'tiger', 'violin', 'guacamole',
             'burrito', 'boat', 'screwdriver', 'turtle', 'spoon', 'cucumber', 'sink', 'lemon', 'trombone', 'snake',
             'tick', 'vase', 'power drill', 'oven', 'electric fan', 'cat', 'bed', 'butterfly', 'bottle', 'cake',
             'saxophone', 'sandwich', 'snowplow', 'water bottle', 'guitar', 'lobster', 'syringe', 'sheep', 'ant',
             'jellyfish', 'pencil sharpener', 'zebra', 'spatula', 'monkey', 'purse', 'hair spray', 'horizontal bar',
             'cup or mug', 'bow tie', 'sports ball', 'wine bottle', 'cocktail shaker', 'ladybug', 'accordion',
             'snowmobile', 'mirror', 'parking meter']

# CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_tokenizer = torch.cat([clip.tokenize(f"a photo of a {c}") for c in CLS_NAMES]).to(device)

# OpenCLIP
opclip_model, _, opclip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k',
                                                                           device=device)
opclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
opclip_text = opclip_tokenizer([f"a photo of a {c}" for c in CLS_NAMES]).to(device)


# CLIP computation
def clip_probs(PIL_Image):
    with torch.no_grad():
        image = clip_preprocess(PIL_Image).unsqueeze(0).to(device)
        logits_per_image, _ = clip_model(image, clip_tokenizer)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs


# OpenCLIP  computation
def open_clip_probs(PIL_Image):
    image = opclip_preprocess(PIL_Image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = opclip_model.encode_image(image)
        text_features = opclip_model.encode_text(opclip_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
    return probs


# filter for high overlap and small area masks
def get_mask_by_samauto(config, image_rgb):
    with torch.no_grad():
        result = mask_generator.generate(image_rgb)
        # clear small area
    del_index = 0
    sorted_result = sorted(result, key=lambda x: x['area'], reverse=True)
    for index, item in enumerate(sorted_result):
        mask = np.array(item["segmentation"], dtype=np.uint8)
        saliency_pixels = np.count_nonzero(mask) / mask.size
        if saliency_pixels <= config.th_miniSm:
            del_index = index
            break
    del sorted_result[del_index:]

    # clear areas with high overlap
    sorted_len = len(sorted_result)
    del_index = set()
    for i in range(sorted_len - 1):
        for j in range(i + 1, sorted_len):
            src1 = np.array(sorted_result[i]["segmentation"], dtype=np.uint8)
            src2 = np.array(sorted_result[j]["segmentation"], dtype=np.uint8)
            result = cv2.bitwise_and(src1, src2)
            out = np.sum(result) / np.sum(src2)
            if out > config.sth_nms: del_index.add(j)

    reverse_del_index = sorted(del_index, reverse=True)
    for index in reverse_del_index:
        del sorted_result[index]
    print("sorted_result number:", len(sorted_result))

    return sorted_result


# get confident masks by sam-everthing mode
def get_salient_masks(config, sam_result, image_bgr, zero_frame, cls_name):
    object_list = []  # mask candidates
    flag = False  # whether there is a reliable mask

    # mask post-processing to filter reliable masks
    for index, mask in enumerate(sam_result):
        box = mask["bbox"]
        x, y, w, h = box
        mask = np.array(mask["segmentation"], dtype=np.uint8)
        if not count_edge_pixels(mask, config.th_salEdge): continue

        mask = np.expand_dims(mask, axis=2)
        mask_image = mask * image_bgr

        crop_mask_image = mask_image[int(y):int(y + h), int(x):int(x + w)]
        PIL_Image = Image.fromarray(crop_mask_image)

        probs = clip_probs(PIL_Image)
        indice = np.argmax(probs[0])
        open_probs = open_clip_probs(PIL_Image)
        open_indice = np.argmax(open_probs[0])
        prob = min(probs[0][indice], open_probs[0][indice])

        if indice == open_indice and CLS_NAMES[indice] == cls_name and prob >= config.sth_clsProb1:
            dict = {}
            dict["index"] = index
            dict["indice"] = indice
            dict["mask"] = mask
            dict["probs"] = prob
            object_list.append(dict)

    if len(object_list) > 0:
        for indx, item in enumerate(sorted(object_list, key=lambda x: x['probs'], reverse=True)):
            probs = item["probs"]
            mask = item["mask"]
            index = item["index"]
            indice = item["indice"]

            if (indx == 0 and probs >= config.sth_clsProb1) or probs >= config.sth_clsProb2:
                zero_frame += mask
                flag = True
                print(index, CLS_NAMES[indice], probs)
    if flag:
        res = morphology(zero_frame, config.morphology_kernal)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        return res, True
    else:
        return zero_frame, False


#
def process(config, img_path, cls_name):
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    zero_frame = np.zeros(image_bgr.shape, dtype=np.uint8)

    sam_result = get_mask_by_samauto(config, image_rgb)
    mask1, flag1 = get_salient_masks(config, sam_result, image_bgr, zero_frame, cls_name)
    if not flag1: return None

    boxes = get_propmt_by_mask(mask1)
    mask2, flag2 = get_mask_by_sambox(config, sam_predictor, image_rgb, boxes)
    if not flag2: return None

    mask_iou = np.sum(cv2.bitwise_and(mask1, mask2)) / (np.sum(cv2.bitwise_or(mask1, mask2)) + 1e-8)
    if mask_iou > config.sth_maskIou:
        return mask1
    else:
        return None


# main function
def main(config):
    images_path = config.input_path
    save_path = config.output_path

    out_image_path = os.path.join(save_path, "image/")
    out_mask_path = os.path.join(save_path, "mask/")
    if not os.path.exists(out_image_path): os.makedirs(out_image_path)
    if not os.path.exists(out_mask_path): os.makedirs(out_mask_path)

    files = os.listdir(images_path)
    tar_len = len(files)
    start_time = time.time()
    for index, image_name in enumerate(files):
        match = re.search(r"([a-zA-Z-\s]+)\d+\.jpg", image_name)
        cls_name = match.group(1).strip() if match else "other"
        print("-" * 20, f"{index}/{tar_len}", image_name, cls_name, "-" * 20)

        png_name = image_name.replace("jpg", "png")

        file_path = os.path.join(images_path, image_name)
        res = process(config, file_path, cls_name)
        if res is not None:
            cv2.imwrite(os.path.join(out_mask_path, png_name), res * 255)
            shutil.copy(file_path, os.path.join(out_image_path, image_name))
    end_time = time.time()
    print(f"{end_time - start_time}秒")

if __name__ == "__main__":
    config = get_arguments()
    print(config)
    main(config)
    print("Done!")

# nohup python -u process/generate_simple_mask.py --gpu="cuda:3" > ./logs/labels/simple1_099-2.log 2>&1 &
