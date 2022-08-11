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
from model.metric import get_confusion_matrix1

def main(config, data_loader):
    logger = config.get_logger('test')
    model = config.init_obj('arch', module_arch)

    loss_fn = config.init_obj('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    total_confusion_matrix = torch.zeros((19, 19))

    with torch.no_grad():
        for i, (data, target, _) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            size = target.size()

            total_confusion_matrix += get_confusion_matrix1(target, output, size=size, num_class=19, ignore=255)
            loss = loss_fn(output, target.to(torch.long))
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
    pos = total_confusion_matrix.sum(1)
    res = total_confusion_matrix.sum(0)
    tp = np.diag(total_confusion_matrix)
        # # print(tp)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # # IoU_array = (tp / torch.clamp(pos + res - tp, min=1.0))
    mean_IoU = (IoU_array).mean()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    logger.info('accuracy: {}'.format(mean_acc))
    logger.info('iou array: {} ||Mean Iou: {}'.format(IoU_array, mean_IoU))
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)



if __name__ == '__main__':
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
    data_dir = 'CityScapes'
    test_data_loader = CityScapesDataLoader(data_dir=data_dir, phase='test', batch_size=1, shuffle=True)

    main(config, test_data_loader)
