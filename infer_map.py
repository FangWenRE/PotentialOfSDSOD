import argparse
import cv2
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
# PyTorch includes
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from methods.maxsum.model import Network
from base.encoder.resnet import resnet
# Custom includes
from utils.data_loader import InferRGB
from process.mask_utils import morphology
from thop import profile


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--input_size', type=int, default=320)
    parser.add_argument('--image_path', type=str, help="the jpg image path")
    parser.add_argument('--save_dir', type=str, help="the png mask save path")
    parser.add_argument('--load_path', type=str, help="model weight path")
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_png(prob_pred, shape, save_path):
    pred = prob_pred.sigmoid().cpu().data.numpy()[0]
    pred = np.clip(np.round(cv2.resize(pred, shape) * 255) / 255.0, 0, 1)
    Image.fromarray((pred * 255)).convert('L').save(save_path)


def main(args):
    encoder = resnet()
    fl = [64, 256, 512, 1024, 2048]
    model = Network(1, encoder=encoder, feat=fl)
    model.load_state_dict(torch.load(args.load_path, map_location="cpu"), strict=True)

    model.to(args.gpu)

    test_data = InferRGB(image_root=args.image_path, return_size=True)
    testloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=8)

    save_dir = args.save_dir
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=os.cpu_count() // 2)
    model.eval()
    with torch.no_grad():
        for sample_batched in tqdm(testloader):
            name, size = sample_batched['name'], sample_batched['size']
            img_shapes = [(b.item(), a.item()) for a, b in zip(size[0], size[1])]

            inputs = sample_batched['image'].clone().detach().to(dtype=torch.float32).to(args.gpu)

            Y = model(inputs)
            preds = Y["final"]
            for i in range(preds.size(0)):
                shape = img_shapes[i]
                save_path = os.path.join(save_dir, name[i].replace("jpg", "png"))
                # save_png(preds[i], shape, save_path)
                pool.apply_async(save_png, (preds[i], shape, save_path))
    pool.close()
    pool.join()

if __name__ == '__main__':
    args = get_arguments()
    main(args)
