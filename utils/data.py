import os, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
from shutil import copyfile, copy
from collections import OrderedDict

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

sub = 'split'  # 'joint'


def copydirs(from_file, to_file):
    if not os.path.exists(to_file):
        os.makedirs(to_file)
    files = os.listdir(from_file)
    for f in files:
        if os.path.isdir(from_file + '/' + f):
            copydirs(from_file + '/' + f, to_file + '/' + f)
        else:
            copy(from_file + '/' + f, to_file + '/' + f)


def OLR(gt_root, name):
    gt_orig = gt_root
    gt_root = os.path.join('./pseudo', 'temp2', name)

    print("Copying labels to temp folder: {}.".format(gt_root))
    copydirs(gt_orig, gt_root)
    print('Using temp labels from {}'.format(gt_root))
    return gt_orig, gt_root


def get_color_list(name, config, phase):
    name_list = []
    if phase == 'train':
        root = config['data_path']
        image_root = os.path.join(root, "image")
        gt_root = os.path.join(root, "mask")
        img_list = os.listdir(image_root)
        print('Using pseudo labels from {}'.format(gt_root))
    else:
        image_root = f"dataset/{name}/image"
        gt_root = f"dataset/{name}/mask"
        img_list = os.listdir(image_root)

    for img_name in img_list:
        img_tag = img_name.split('.')[0]

        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        name_list.append(tag_dict)

    return name_list


def get_rgbd_list(name, config, phase):
    name_list = []
    image_root = os.path.join(config['data_path'], 'RGBD/{}/image'.format(name))
    dep_root = os.path.join(config['data_path'], 'RGBD/{}/depth'.format(name))

    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/d-split/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))

        if config['olr']:
            gt_orig, gt_root = OLR(gt_root, name)
    else:
        gt_root = os.path.join(config['data_path'], 'RGBD/{}/mask'.format(name))

    img_list = os.listdir(image_root)
    for img_name in img_list:
        img_tag = img_name.split('.')[0]

        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        tag_dict['dep'] = os.path.join(dep_root, img_tag + '.png')
        name_list.append(tag_dict)

    return name_list


def get_rgbt_list(name, config, phase):
    name_list = []
    image_root = os.path.join(config['data_path'], 'RGBT/{}/RGB'.format(name))
    th_root = os.path.join(config['data_path'], 'RGBT/{}/T'.format(name))

    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/t-split/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))

        if config['olr']:
            gt_orig, gt_root = OLR(gt_root, name)
    else:
        gt_root = os.path.join(config['data_path'], 'RGBT/{}/GT'.format(name))

    img_list = os.listdir(image_root)
    for img_name in img_list:
        img_tag = img_name.split('.')[0]

        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        tag_dict['th'] = os.path.join(th_root, img_tag + '.jpg')
        name_list.append(tag_dict)

    return name_list


def get_frame_list(name, config, phase):
    name_list = []

    base_path = os.path.join(config['data_path'], 'vsod', name)
    videos = os.listdir(os.path.join(base_path, 'JPEGImages'))

    if config['stage'] > 1 and phase == 'train':
        gt_base = './pseudo/o-joint/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_base))

        if config['olr']:
            gt_orig, gt_base = OLR(gt_base, name)
    else:
        gt_base = os.path.join(base_path, 'Annotations')

    for video in videos:
        image_root = os.path.join(base_path, 'JPEGImages', video)
        gt_root = os.path.join(gt_base, video)
        of_root = os.path.join(base_path, 'optical', video)

        img_list = os.listdir(image_root)
        img_list = sorted(img_list)
        if phase == 'train' and config['stage'] == 1 and 'select' in video:
            img_list = img_list[::5]

        for img_name in img_list:
            img_tag = img_name.split('.')[0]

            tag_dict = {}
            tag_dict['rgb'] = os.path.join(image_root, img_name)
            tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
            tag_dict['of'] = os.path.join(of_root, img_tag + '.jpg')
            name_list.append(tag_dict)

    return name_list


def get_train_image_list(config):
    phase = 'train'
    image_list = get_color_list(None, config, phase)

    print('Load {} images for training.'.format(len(image_list)))
    return image_list


def get_test_list(config=None, phase='train'):
    test_dataset = OrderedDict()
    if phase == "train":
        test_list = ['DUTS-TE']
    else:
        test_list = ['PASCAL-S', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'DUTS-TE']  # , 'SOD'

    for test_set in test_list:
        test_dataset[test_set] = Test_Dataset(test_set, config)

    return test_dataset


def get_loader(config):
    dataset = Train_Dataset(config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader


def read_modality(sub, sample_path, flip, img_size):
    if sub in sample_path.keys():
        m_name = sample_path[sub]
        modal = Image.open(m_name).convert('RGB')
        modal = modal.resize((img_size, img_size))
        modal = np.array(modal).astype(np.float32) / 255.
        if flip:
            modal = modal[:, ::-1].copy()
        modal = modal.transpose((2, 0, 1))
    else:
        modal = np.zeros((3, img_size, img_size)).astype(np.float32)

    return modal


def get_dict(txt_path):
    name_iou = {}
    with open(txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            name = parts[0]
            iou = float(parts[1])
            name_iou[name] = iou
    return name_iou


class Train_Dataset(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.image_list = get_train_image_list(config)
        self.size = len(self.image_list)

    def __getitem__(self, index):
        sample_path = self.image_list[index]
        img_name = sample_path['rgb']
        gt_name = sample_path['gt']

        image = Image.open(img_name).convert('RGB')
        gt = Image.open(gt_name).convert('L')

        img_size = self.config['size']
        image = image.resize((img_size, img_size))
        gt = gt.resize((img_size, img_size), Image.NEAREST)

        image = np.array(image).astype(np.float32)
        gt = np.array(gt)

        flip = random.random() > 0.5
        if flip:
            image = image[:, ::-1].copy()
            gt = gt[:, ::-1].copy()

        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)
        gt = np.where(gt >= 0.5, 1, 0)
        gt = np.expand_dims(gt, axis=0)
        out_dict = {'image': image, 'gt': gt, 'name': gt_name, 'flip': flip}
        return out_dict

    def __len__(self):
        return self.size


class Test_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        self.image_list = get_color_list(name, config, 'test')

        self.set_name = name
        self.size = len(self.image_list)

    def load_data(self, index):
        sample_path = self.image_list[index]
        img_name = sample_path['rgb']
        gt_name = sample_path['gt']

        image = Image.open(img_name).convert('RGB')
        image = image.resize((self.config['size'], self.config['size']))
        image = np.array(image).astype(np.float32)
        gt = Image.open(gt_name).convert('L')
        img_size = self.config['size']

        img_pads = img_name.split('/')
        name = '/'.join(img_pads[img_pads.index(self.set_name) + 2:])

        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)

        out_dict = {'image': image, 'gt': gt, 'name': name}
        return out_dict
