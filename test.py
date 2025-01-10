import sys
import torch
import time
import os
import cv2
from PIL import Image
from utils.util import *
import numpy as np
from progress.bar import Bar
from utils.data import Test_Dataset, get_test_list
from base.framework_factory import load_framework
from utils.metric import *


def test_model(model, test_sets, config, epoch=None, MAX_F=0):
    model.eval()
    st = time.time()
    for set_name, test_set in test_sets.items():
        titer = test_set.size
        MR = MetricRecorder(titer)
        ious = []
        dises = []

        test_bar = Bar('Dataset {}:'.format(set_name), max=titer)
        for j in range(titer):
            pack = test_set.load_data(j)

            images = torch.tensor(pack['image']).float()
            gt = pack['gt']
            name = pack['name']
            images = images.cuda()
            out_shape = gt.shape[-2:]
            Y = model(images)

            pred = Y['final'].sigmoid().cpu().data.numpy()[0, 0]
            gt = (gt > 0.5).astype(float)

            # CRF the pred if you need
            if config['crf']:
                pred = (pred * 255).astype(np.uint8)
                thre, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_OTSU)
                pred, gt = normalize_pil(pred, gt)

                mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
                std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
                orig_img = images[0].cpu().numpy().transpose(1, 2, 0)
                orig_img = ((orig_img * std + mean) * 255.).astype(np.uint8)

                pred = (pred > 0.5).astype(np.uint8)
                pred = crf_inference_label(orig_img, pred)
                pred = cv2.medianBlur(pred.astype(np.uint8), 7)

            pred = np.clip(np.round(cv2.resize(pred, out_shape[::-1]) * 255) / 255., 0, 1)
            MR.update(pre=pred, gt=gt)
            gt = (gt > 0.5).astype(np.float32)
            iou = cal_iou(pred, gt)
            ious.append(iou)
            dis = cal_dis(pred, gt)
            dises.append(dis)

            # save predictions
            if config['save']:
                os.makedirs("SM", exist_ok=True)
                fnl_folder = os.path.join('./SM', config["save_tar"], set_name)
                check_path(fnl_folder)
                im_path = os.path.join(fnl_folder, name.split('.')[0] + '.png')
                Image.fromarray((pred * 255)).convert('L').save(im_path)

            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()

        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        print("\n", config["save_tar"], set_name)
        print('MAX-F: {}, MAE: {:.3f}, EM: {}, IOU: {:.3f}, dis: {:.3f}.'.format(maxf, round(mae, 3), em, np.mean(ious),
                                                                                 np.mean(dises)))

    if epoch is not None:
        if MAX_F < maxf:
            weight_path = os.path.join(config['weight_path'], 'best.pth')
            torch.save(model.state_dict(), weight_path)
            MAX_F = maxf
            print(f"This epoch{epoch} is a best model!")
        # else:
        # weight_path = os.path.join(config['weight_path'], '{}_{}.pth'.format(config['model_name'], epoch))
        # torch.save(model.state_dict(), weight_path)    
    print('Test using time: {}.'.format(round(time.time() - st, 3)))
    return MAX_F


def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return

    config, model, _, _, _, saver = load_framework(net_name)
    config['net_name'] = net_name

    if config['crf']:
        config['orig_size'] = True

    if config['weight'] != '':
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
    else:
        print('No weight file provide!')

    test_sets = get_test_list(config, phase="test")
    model = model.cuda()
    test_model(model, test_sets, config)


if __name__ == "__main__":
    main()
