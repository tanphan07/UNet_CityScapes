import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.network as module_arch
from parse_config import ConfigParser
import numpy as np
from utils.util import decode_segmap
import matplotlib.pyplot as plt
from data_loader import CityScapesDataLoader
import cv2
from torch.nn import functional as F
from tqdm import tqdm


color_map = [(128, 64, 128),
             (244, 35, 232),
             (70, 70, 70),
             (102, 102, 156),
             (190, 153, 153),
             (153, 153, 153),
             (250, 170, 30),
             (220, 220, 0),
             (107, 142, 35),
             (152, 251, 152),
             (70, 130, 180),
             (220, 20, 60),
             (255, 0, 0),
             (0, 0, 142),
             (0, 0, 70),
             (0, 60, 100),
             (0, 80, 100),
             (0, 0, 230),
             (119, 11, 32)]


def main(config, img, anno, name):
    model = config.init_obj('arch', module_arch)
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    # size = anno.size()
    img = img.to(device)
    out_put = model(img)
    out_put = F.interpolate(
        input=out_put, size=(1024, 2048),
        mode='bilinear', align_corners=True
    )
    # for i, metric in enumerate(metrics):
    #     print('{}: {:.6f}'.format(metric.__name__, metric(out_put, anno)))
    out_put = torch.argmax(out_put, dim=1)

    for jj in range(img.size()[0]):
        img = test_img.cpu().numpy()
        show_output = out_put.cpu().numpy()
        gt = anno.numpy()

        tmp = np.array(show_output[jj]).astype(np.uint8)

        tmp1 = np.array(gt[jj]).astype(np.uint8)
        ignore_index = np.where(tmp1 == 255)
        tmp[ignore_index] = 255
        sv_img = np.zeros((1024, 2048, 3))
        for i, color in enumerate(color_map):
            for j in range(3):
                sv_img[:, :, j][tmp == i] = color_map[i][j]

        true_name = name[0].split('/')[-1]
        cv2.imwrite('/home/tanpv/workspace/Segment_SDC/out_images/' + 'Unet_' + true_name, sv_img)


if __name__ == '__main__':
    data_dir = 'CityScapes'
    test_data_loader = CityScapesDataLoader(data_dir=data_dir, phase='test', batch_size=1, shuffle=True)
    # test_img, anno, name = next(iter(test_data_loader))

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of training session ')
    config = ConfigParser.from_args(args)
    for test_img, anno, name in tqdm(test_data_loader, total=len(test_data_loader)):
        main(config, test_img, anno, name)
