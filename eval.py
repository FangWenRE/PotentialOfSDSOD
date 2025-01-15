from progress.bar import Bar
import os
from PIL import Image
from utils.util import *
import numpy as np
import argparse

from utils.data import *
from utils.metric import *


# python3 eval.py  --sm_path=""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sm_path', default='SM/*2stage_batch12',
                        help='The saliency map path is inferred by the trained model')
    params = parser.parse_args()
    config = vars(params)
    config['orig_size'] = True
    config['size'] = 320
    test_sets = get_test_list(config=config, phase='test')

    for set_name, test_set in test_sets.items():
        mask_path = os.path.join(config['sm_path'], set_name)

        if not os.path.exists(mask_path):
            print('{} not exists!!!!!'.format(mask_path))
            continue

        titer = test_set.size
        MR = MetricRecorder(titer)
        test_bar = Bar('{}'.format(set_name), max=titer)

        for j in range(titer):
            sample_dict = test_set.load_data(j)
            gt = sample_dict['gt']
            name = sample_dict['name']
            name = name.split('.')[0]
            pred = Image.open(os.path.join(mask_path, f"{name}.png")).convert('L')
            out_shape = gt.shape
            pred = np.array(pred.resize((out_shape[::-1])))

            pred, gt = normalize_pil(pred, gt)
            MR.update(pre=pred, gt=gt)
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()

        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        print('\nMax-F: {}, MAE: {}, Maen-F: {}, Fbw: {},  SM: {}, EM: {}.'.format(maxf, mae, meanf, wfm, sm, em))


if __name__ == "__main__":
    main()
