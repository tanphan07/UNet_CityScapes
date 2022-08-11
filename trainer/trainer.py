import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, report_eta
from tqdm import tqdm
import time
import datetime
from model import adjust_optim
import torch.nn.functional as F
from model.metric import get_confusion_matrix1


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.log_step = int(train_data_loader.batch_size / 2)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.total_iter = self.epochs * self.len_epoch
        # print(self.total_iter)
        self.init_lr = optimizer.param_groups[0]['lr']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        t_start = time.time()
        # total_loss = 0
        total_confusion_matrix = torch.zeros((19, 19))

        for batch_idx, (data, target) in enumerate(self.train_data_loader):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            # print(output.shape)
            loss = self.criterion(output, target)
            # losses = loss.mean()
            loss.backward()
            self.optimizer.step()

            size = target.size()

            total_confusion_matrix += get_confusion_matrix1(target, output, size=size, num_class=19, ignore=255)
            # total_confusion_matrix += get_confusion_matrix1(target, output, number_classes=19)
            step = (epoch - 1) * self.len_epoch + batch_idx
            self.writer.set_step(step)
            self.train_metrics.update('loss', loss.item(), step)
            #
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), step)
            # print(self.lr_scheduler.get_last_lr())
            lr = adjust_optim(optimizer=self.optimizer, current_iter=step, total_iters=self.total_iter, factor=0.9,
                              init_lr=self.init_lr)
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}  Lr: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(), lr))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            # total_loss += losses.item()
            if batch_idx == self.len_epoch:
                break

        pos = total_confusion_matrix.sum(1)
        res = total_confusion_matrix.sum(0)
        tp = np.diag(total_confusion_matrix)
        # # print(tp)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # # IoU_array = (tp / torch.clamp(pos + res - tp, min=1.0))
        mean_IoU = (IoU_array).mean()
        #
        self.writer.add_scalar('train_mIoU_per_epoch', mean_IoU, epoch)

        log = self.train_metrics.result()
        #
        self.writer.add_scalar('train_loss_per_epoch', log['loss'], epoch)
        self.writer.add_scalar('train_accuracy_per_epoch', log['accuracy'], epoch)

        t_end = time.time()

        self.logger.debug('Training time: {}'.format(str(datetime.timedelta(seconds=(t_end - t_start)))))
        if self.do_validation:
            val_log, iou_array, miou = self._valid_epoch(epoch)

            self.writer.add_scalar('valid_loss_per_epoch', val_log['loss'], epoch)
            self.writer.add_scalar('valid_accuracy_per_epoch', val_log['accuracy'], epoch)

            log.update(**{'val_' + k: v for k, v in val_log.items()})
            self.logger.debug('iou array: {} ||Mean Iou: {}'.format(iou_array, miou))

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        total_confusion_matrix = torch.zeros((19, 19))
        # total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                size = target.size()
                # h, w = target.size(1), target.size(2)
                #
                # ph, pw = output[0].size(2), output[0].size(3)
                # if ph != h or pw != w:
                #     for i in range(len(output)):
                #         output[i] = F.interpolate(output[i], size=(
                #             h, w), mode='bilinear', align_corners=True)

                loss = self.criterion(output, target)
                # losses = loss.mean()

                # total_loss += losses.item()
                total_confusion_matrix += get_confusion_matrix1(target, output, size=size, num_class=19, ignore=255)
                # total_confusion_matrix += get_confusion_matrix1(target, output, number_classes=19)

                val_step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                self.writer.set_step(val_step, 'valid')
                self.valid_metrics.update('loss', loss.item(), val_step)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), val_step)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        pos = total_confusion_matrix.sum(1)
        res = total_confusion_matrix.sum(0)
        tp = np.diag(total_confusion_matrix)
        # print(tp)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # IoU_array = (tp / torch.clamp(pos + res - tp, min=1.0))
        mean_IoU = (IoU_array).mean()
        #
        self.writer.add_scalar('valid_mIoU_per_epoch', mean_IoU, epoch)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), IoU_array, mean_IoU

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
