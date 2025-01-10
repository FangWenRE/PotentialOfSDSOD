import torch
from torch import nn
from torch.nn import functional as F
from utils.util import *

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

def Loss(preds, targets):
    bce = nn.BCELoss()
    total_loss = 0
    loss_list = []
    for name, pred, target in zip(["sal","edge"], preds['sal'], targets):
        pred = F.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        target = target.gt(0.5).float()
        pred = torch.sigmoid(pred)
        if name == "sal":
            loss = IOU(pred, target)
        else:
            loss = bce(pred, target)
        loss_list.append(loss)
        total_loss += loss

    return total_loss, loss_list
