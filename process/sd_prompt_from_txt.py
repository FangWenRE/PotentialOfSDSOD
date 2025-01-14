import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from PIL import Image

def get_pipe(device = "cuda:0"):
    model_id = "/data/Data/stable-diffusion-2-1"
    # model_id = "/data/Data/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.safety_checker = lambda images, clip_input: (images, False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def get_inputs(prompt, batch_size=1):
    # generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20
    guidance_scale = 7.5
    line = {"prompt": prompts, "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps}
    return line

def get_prompts(txt_path):
    with open(txt_path) as f:
        lines = list(filter(lambda x:len(x)>1, f.readlines()))
    return lines

if __name__ == '__main__':
    device = "cuda:0"
    prompts_path = "class_prompts/one_adj_prompts"   
    
    coco_name80 = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'computer mouse', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'ski', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']
    cls_name176 = ['house','centipede', 'tape player', 'piano', 'crutch', 'desk', 'fox', 'ruler', 'helmet', 'french horn', 'salt or pepper shaker', 'computer keyboard', 'swine', 'drum', 'ray', 'snail', 'porcupine', 'microphone', 'bathing cap', 'pencil box', 'mouse', 'starfish', 'artichoke', 'harmonica', 'strawberry', 'nail', 'baseball', 'punching bag', 'digital clock', 'trumpet', 'rugby ball', 'cart', 'bee', 'swimming trunks', 'waffle iron', 'otter', 'plate rack', 'harp', 'golf ball', 'frying pan', 'table', 'flute', 'seal', 'washer', 'bell pepper', 'ladle', 'bagel', 'oboe', 'domestic cat', 'hamster', 'basketball', 'rabbit', 'head cabbage', 'lizard', 'chime', 'cello', 'coffee maker', 'popsicle', 'corkscrew', 'rubber eraser', 'tennis ball', 'bookshelf', 'baby bed', 'goldfish', 'pineapple', 'squirrel', 'stove', 'balance beam', 'binder', 'miniskirt', 'remote control', 'sunglasses', 'bow', 'beaker', 'lamp', 'brassiere', 'diaper', 'mushroom', 'hat with a wide brim', 'pretzel', 'stretcher', 'window', 'flower pot', 'plastic bag', 'sofa', 'perfume', 'lion', 'vacuum', 'red panda', 'camel', 'face powder', 'whale', 'unicycle', 'printer', 'cattle', 'hamburger', 'fig', 'dumbbell', 'dishwasher', 'ping-pong ball', 'croquet ball', 'neck brace', 'koala bear', 'iPod', 'filing cabinet', 'volleyball', 'scorpion', 'antelope', 'chain saw', 'watercraft', 'frog', 'stethoscope', 'hammer', 'axe', 'giant panda', 'isopod', 'racket', 'band aid', 'soccer ball', 'milk can', 'banjo', 'dragonfly', 'pitcher', 'puck', 'pomegranate', 'hotdog', 'skunk', 'strainer', 'can opener', 'golfcart', 'soap dispenser', 'hair dryer', 'door', 'lipstick', 'blender', 'maraca', 'cream', 'maillot', 'armadillo', 'hippopotamus', 'tiger', 'violin', 'guacamole', 'burrito', 'screwdriver', 'turtle', 'cucumber', 'lemon', 'trombone', 'snake', 'tick', 'power drill', 'electric fan', 'butterfly', 'saxophone', 'snowplow', 'water bottle', 'guitar', 'lobster', 'syringe', 'ant', 'jellyfish', 'pencil sharpener', 'spatula', 'monkey', 'purse', 'hair spray', 'horizontal bar', 'cup or mug', 'bow tie', 'wine bottle', 'cocktail shaker', 'ladybug', 'accordion', 'snowmobile', 'mirror']
    
    save_path = ""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pipe = get_pipe(device)
   
    for file_name in os.listdir(prompts_path):
        num = 0
        class_name = file_name[:-4]
        if class_name in coco_name80:
            batch_size = 3
        else:
            batch_size = 2
        
        prompts = get_prompts(os.path.join(prompts_path, file_name))
        for prompt in prompts:
            # prompt = 'A professional photo of ' + str(prompt).lower()
            images = pipe(**get_inputs(prompt.strip(), batch_size=batch_size)).images
            for image in images:
                num += 1
                image.save(os.path.join(save_path, f"{class_name}{num}.jpg"))
        print(f"{file_name} has Done!")