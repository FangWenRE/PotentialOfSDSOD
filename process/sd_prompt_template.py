import torch, os, random
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def get_pipe(device = "cuda:0"):
    # model_id = "/data/Data/stable-diffusion-2-1"
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.safety_checker = lambda images, clip_input: (images, False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_inputs(prompt, batch_size=1):
    # generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20
    guidance_scale = 7.5
    line = {"prompt": prompts, "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps}
    return line


if __name__ == '__main__':
    num = 0
    batch_size= 5
    device = "cuda:0"
   
    save_path = "DataSet/APhotoOf"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pipe = get_pipe(device)

    '''
    for key in ["man","woman","boy","girl"]:for express in ["smile", "serious", "calm", "sad"]:prompt = f"A professional normal photo of one healthy {key} looking forward with a {express} face, taken with an 80mm lens,suitable environment"
    A professional normal photo of {house/hut} with empty backgraound, taken with an 35mm lens,suitable environment
    prompt = f"a professional photo of only one {key} in suitable environment, taken with a 30mm lens, non-cutaway photo"
    prompt = f"A professional and realistic photo of one {key} with backgraound such as {a},{b},{c},{d},etc, non-cutaway photo"
    '''

    coco_name80 = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'computer mouse', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'ski', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']
    cls_name176 = ['house','centipede', 'tape player', 'piano', 'crutch', 'desk', 'fox', 'ruler', 'helmet', 'french horn', 'salt or pepper shaker', 'computer keyboard', 'swine', 'drum', 'ray', 'snail', 'porcupine', 'microphone', 'bathing cap', 'pencil box', 'mouse', 'starfish', 'artichoke', 'harmonica', 'strawberry', 'nail', 'baseball', 'punching bag', 'digital clock', 'trumpet', 'rugby ball', 'cart', 'bee', 'swimming trunks', 'waffle iron', 'otter', 'plate rack', 'harp', 'golf ball', 'frying pan', 'table', 'flute', 'seal', 'washer', 'bell pepper', 'ladle', 'bagel', 'oboe', 'domestic cat', 'hamster', 'basketball', 'rabbit', 'head cabbage', 'lizard', 'chime', 'cello', 'coffee maker', 'popsicle', 'corkscrew', 'rubber eraser', 'tennis ball', 'bookshelf', 'baby bed', 'goldfish', 'pineapple', 'squirrel', 'stove', 'balance beam', 'binder', 'miniskirt', 'remote control', 'sunglasses', 'bow', 'beaker', 'lamp', 'brassiere', 'diaper', 'mushroom', 'hat with a wide brim', 'pretzel', 'stretcher', 'window', 'flower pot', 'plastic bag', 'sofa', 'perfume', 'lion', 'vacuum', 'red panda', 'camel', 'face powder', 'whale', 'unicycle', 'printer', 'cattle', 'hamburger', 'fig', 'dumbbell', 'dishwasher', 'ping-pong ball', 'croquet ball', 'neck brace', 'koala bear', 'iPod', 'filing cabinet', 'volleyball', 'scorpion', 'antelope', 'chain saw', 'watercraft', 'frog', 'stethoscope', 'hammer', 'axe', 'giant panda', 'isopod', 'racket', 'band aid', 'soccer ball', 'milk can', 'banjo', 'dragonfly', 'pitcher', 'puck', 'pomegranate', 'hotdog', 'skunk', 'strainer', 'can opener', 'golfcart', 'soap dispenser', 'hair dryer', 'door', 'lipstick', 'blender', 'maraca', 'cream', 'maillot', 'armadillo', 'hippopotamus', 'tiger', 'violin', 'guacamole', 'burrito', 'screwdriver', 'turtle', 'cucumber', 'lemon', 'trombone', 'snake', 'tick', 'power drill', 'electric fan', 'butterfly', 'saxophone', 'snowplow', 'water bottle', 'guitar', 'lobster', 'syringe', 'ant', 'jellyfish', 'pencil sharpener', 'spatula', 'monkey', 'purse', 'hair spray', 'horizontal bar', 'cup or mug', 'bow tie', 'wine bottle', 'cocktail shaker', 'ladybug', 'accordion', 'snowmobile', 'mirror']
    
    cls_name = [coco_name80, cls_name176]
    bg_words =['sky', 'clouds', 'sun', 'moon', 'stars', 'rain', 'thunder', 'lightning', 'wind', 'breeze', 'hurricane', 'tornado', 'typhoon', 'hail', 'snow', 'blizzard', 'fog', 'mist', 'dew', 'rainbow', 'sunrise', 'sunset', 'horizon', 'landscape', 'scenery', 'nature', 'grass', 'meadow', 'field', 'forest', 'woods', 'jungle', 'desert', 'sand', 'dunes', 'mountain', 'hill', 'valley', 'canyon', 'plateau', 'island', 'beach', 'ocean', 'sea', 'waves', 'coast', 'shore', 'river', 'stream', 'waterfall', 'lake', 'pond', 'pool', 'brook', 'bridge', 'path', 'road', 'street', 'alley', 'highway', 'railway', 'tracks', 'station', 'tunnel', 'underpass', 'overpass', 'airport', 'runway', 'harbor', 'port', 'wharf', 'pier', 'dock', 'building', 'skyscraper', 'tower', 'chimney', 'roof', 'glass', 'window', 'door', 'fence', 'gate', 'lawn', 'garden', 'pathway', 'driveway', 'sidewalk', 'streetlamp', 'bench', 'picnic', 'table', 'chair', 'umbrella', 'flag',"tree","grass"]
    
    for index, keys in enumerate(cls_name):
        if index == 0:
            iter = 14
        else:
            iter = 8
        for key in keys:
            for _ in range(iter):
                a, b, c, d = random.sample(bg_words, 4)
                prompt = f"a professional photo of one {key} with suitable background such as {a},{b},{c},{d},etc."
                images = pipe(**get_inputs(prompt, batch_size=batch_size)).images
                for image in images:
                    num += 1
                    image.save(os.path.join(save_path, f"{key}{num}.jpg"))
            print(f"total num is {num}")       
       