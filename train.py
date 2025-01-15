import sys

import cv2
import torch
from math import exp
from tqdm import trange
from progress.bar import Bar
from utils.util import *
from utils.data import Train_Dataset, Test_Dataset, get_loader, get_test_list
from test import test_model
from torch.nn import utils
from base.framework_factory import load_framework
from utils.util import label_edge_prediction, seed_everything

torch.set_printoptions(precision=5)

def main():
    # Loading model
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
        
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    config['net_name'] = net_name

    if config['weight'] != '':
        print('Load weights from: {}.'.format(config['weight']))
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'), strict=False)

    train_loader = get_loader(config)
    test_sets = get_test_list(config, "train")
    
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    ave_batch = config['ave_batch']
    batch_idx = 0
    model.zero_grad()
    MAX_F = 0
    for epoch in trange(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()

        bar = Bar('{}|epoch {:2}:'.format(net_name, epoch), max=num_iter)

        loss_count = 0
        optim.zero_grad()
        
        fin_lr = 0.2
        for i, pack in enumerate(train_loader, start=1):
            cur_it = i + (epoch-1) * num_iter
            total_it = num_epoch * num_iter
            itr = (1 - cur_it / total_it) * (1 - fin_lr) + fin_lr
            mul = itr
            
            optim.param_groups[0]['lr'] = config['lr'] * mul * 0.1
            optim.param_groups[1]['lr'] = config['lr'] * mul

            images = pack['image'].float()
            gts = pack['gt'].float()
            gt_names = pack['name']
            flips = pack['flip']
            images, gts = images.cuda(), gts.cuda()

            Y = model(images)
            edge_gt = label_edge_prediction(gts)
            loss, loss_item = model_loss(Y, [gts, edge_gt])

            loss_count += loss.data
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            lrs = ','.join([format(param['lr'], ".2e") for param in optim.param_groups])
            lossi = f"{loss_item[0]:.3f},{loss_item[1]:.3f}"
            Bar.suffix = '{:4}/{:4} | loss: {:1.3f}, lossi:{}, LRs: [{}],'.format(i, num_iter, float(loss_count / i), lossi, lrs)
            bar.next()
            
            if epoch > 1 and config['olr']:
                lamda = config['resdual']
                for gt_path, pred, gt, flip in zip(gt_names, torch.sigmoid(Y['final'].detach()), gts, flips):
                    if flip:
                        pred = pred.flip(2)
                    new_gt = (pred * (1 - lamda)).cpu().numpy().transpose(1, 2, 0)
                    cv2.imwrite(gt_path, new_gt * 255)
                
        sche.step()
        bar.finish()
        torch.cuda.empty_cache()
        MAX_F = test_model(model, test_sets, config, epoch, MAX_F=MAX_F)

if __name__ == "__main__":
    seed_everything(1024)
    main()
