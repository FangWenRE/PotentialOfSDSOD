import torch
from torch import nn
from torch.nn import functional as F

mode = 'bilinear'  # 'nearest' #


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        bias = False if bn else False
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def up_conv(cin, cout, up=True):
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(cout // 2, cout)
    yield nn.ReLU(inplace=True)
    if up: yield nn.Upsample(scale_factor=2, mode='bilinear')


def local_conv(cin, cout):
    yield nn.Conv2d(cin, cout * 2, 3, padding=1)
    yield nn.GroupNorm(cout, cout * 2)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')

    yield nn.Conv2d(cout * 2, cout, 3, padding=1)
    yield nn.GroupNorm(cout // 2, cout)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')


class FFM(nn.Module):
    def __init__(self, tar_feat):
        super(FFM, self).__init__()
        self.gconv = nn.Sequential(*list(up_conv(tar_feat, tar_feat, False)))
        self.res_conv = nn.Conv2d(tar_feat, tar_feat, 3, padding=1)

    def forward(self, xs, glob_x):
        loc_x1 = xs[0]
        glob_x0 = nn.functional.interpolate(self.gconv(glob_x), size=loc_x1.size()[2:], mode=mode)
        loc_x2 = nn.functional.interpolate(xs[1], size=loc_x1.size()[2:], mode=mode)

        max_values, _ = torch.max(torch.stack([loc_x1, glob_x0, loc_x2]), dim=0)
        res = self.res_conv(max_values)
        return res


class EEM(nn.Module):
    def __init__(self, tar_feat):
        super(EEM, self).__init__()
        self.gconv = nn.Sequential(*list(up_conv(tar_feat, tar_feat, False)))
        self.res_conv = nn.Conv2d(tar_feat, tar_feat, 3, padding=1)
        self.fuse = nn.Conv2d(tar_feat * 2, tar_feat, 3, padding=1)

    def forward(self, xs, glob_x):
        loc_x1 = xs[0]
        glob_x0 = nn.functional.interpolate(self.gconv(glob_x), size=loc_x1.size()[2:], mode=mode)
        loc_x2 = nn.functional.interpolate(xs[1], size=loc_x1.size()[2:], mode=mode)

        local_mean = (loc_x1 + loc_x2) / 2
        res = torch.sigmoid(self.res_conv(local_mean - glob_x0))
        edge = local_mean * res
        loc_x = self.fuse(torch.cat([edge, glob_x0], dim=1))
        return loc_x, edge


class decoder(nn.Module):
    def __init__(self, feat):
        super(decoder, self).__init__()

        self.adapter0 = nn.Sequential(*list(up_conv(feat[0], feat[0] * 3, False)))
        self.adapter1 = nn.Sequential(*list(up_conv(feat[1], feat[0] * 3, False)))
        self.adapter2 = nn.Sequential(*list(up_conv(feat[2], feat[0] * 3, False)))
        self.adapter3 = nn.Sequential(*list(up_conv(feat[3], feat[0] * 3, False)))
        self.adapter4 = nn.Sequential(*list(up_conv(feat[4], feat[0] * 3, False)))

        self.region = FFM(feat[0] * 3)
        self.local = EEM(feat[0] * 3)

        self.gb_conv = nn.Sequential(*list(local_conv(feat[0] * 3, feat[0] * 3)))

        self.edge_pred = nn.Sequential(*list(up_conv(feat[0] * 3, feat[0] * 3, False)), nn.Conv2d(feat[0] * 3, 1, 1))
        self.sal_pred = nn.Sequential(*list(up_conv(feat[0] * 3, feat[0] * 3, False)), nn.Conv2d(feat[0] * 3, 1, 1))

    def forward(self, xs, x_size):
        xs[0] = self.adapter0(xs[0])
        xs[1] = self.adapter1(xs[1])
        xs[2] = self.adapter2(xs[2])
        xs[3] = self.adapter3(xs[3])
        xs[4] = self.adapter4(xs[4])

        glob_x = xs[4]
        reg_x = self.region(xs[2:4], glob_x)

        glob_x = self.gb_conv(glob_x)
        loc_x, edge = self.local(xs[0:2], glob_x)

        edge_pred = self.edge_pred(edge)
        edge_pred = F.interpolate(edge_pred, size=x_size, mode='bilinear')

        reg_x = F.interpolate(reg_x, size=xs[0].size()[2:], mode=mode)
        pred = self.sal_pred(loc_x * reg_x)
        pred = F.interpolate(pred, size=x_size, mode='bilinear')

        OutDict = {}
        OutDict['sal'] = [pred, edge_pred]
        OutDict['final'] = pred
        return OutDict


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        self.encoder = encoder
        self.decoder = decoder(feat)
        # self._init_weight()

    def _init_weight(self):
        print(">>> init_weight")
        for m in self.decoder.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = x[0]
        x_size = x.size()[2:]
        xs = self.encoder(x)
        out = self.decoder(xs, x_size)
        return out
