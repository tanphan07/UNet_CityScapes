import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from pytorch_toolbelt import losses as L


def nll_loss(output, target):
    return F.nll_loss(output, target)


# class DiceLoss(nn.Module):
#     def __init__(self, num_classes=21):
#         super(DiceLoss, self).__init__()
#         self.num_classes = num_classes
#
#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=1).exp()
#         # print(torch.unique(target))
#         target = F.one_hot(target.to(torch.int64), num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
#         smooth = 0.0001
#         iflat = pred.contiguous().view(-1)
#         tflat = target.contiguous().view(-1)
#         intersection = (iflat * tflat).sum()
#         A_sum = torch.sum(iflat * iflat)
#         B_sum = torch.sum(tflat * tflat)
#         return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.classes = np.arange(num_classes)
        self.criterion = L.DiceLoss(mode='multiclass', classes=self.classes, ignore_index=ignore_index)

    def forward(self, pred, target):
        return self.criterion(pred, target)


class DiceLossPSP(nn.Module):
    def __init__(self, aux_weight=0.4, num_classes=19):
        super(DiceLossPSP, self).__init__()
        self.aux_weight = aux_weight
        self.main_loss = DiceLoss(num_classes)
        self.aux_loss = DiceLoss(num_classes)

    def forward(self, pred, target):
        return (1 - self.aux_weight / 2) * self.main_loss(pred[0], target) + self.aux_weight / 2 * self.aux_loss(
            pred[1], target)


class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        targets = targets.type(torch.LongTensor).to(torch.device('cuda'))
        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

        return (1 - self.aux_weight) * loss + self.aux_weight * loss_aux


class SegmentationLosses(nn.Module):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=True, mode='ce'):
        super(SegmentationLosses, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.mode = mode

    def cross_entropy_loss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def focal_loss(self, logit, target, gamma=1, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def forward(self, outputs, target):
        if self.mode == 'ce':
            return self.cross_entropy_loss(outputs, target)
        elif self.mode == 'focal':
            return self.focal_loss(outputs, target)
        else:
            raise NotImplementedError


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data.to(torch.int64), 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1., ignore_index=255):
#         super(DiceLoss, self).__init__()
#         self.ignore_index = ignore_index
#         self.smooth = smooth
#
#     def forward(self, output, target):
#         if self.ignore_index not in range(int(target.min()), int(target.max())):
#             if (target == self.ignore_index).sum() > 0:
#                 target[(target == self.ignore_index).to(torch.int64)] = target.min()
#         target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
#         output = output.log_softmax(dim=1).exp()
#         output_flat = output.contiguous().view(-1)
#         target_flat = target.contiguous().view(-1)
#         intersection = (output_flat * target_flat).sum()
#         loss = 1 - ((2. * intersection + self.smooth) /
#                     (output_flat.sum() + target_flat.sum() + self.smooth))
#         return loss
#

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss(num_classes=19)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target.long())
        dice_loss = self.dice(output, target.long())
        return CE_loss + dice_loss


# Loss for RegNet
class BootstrappedCE(nn.Module):
    def __init__(self, min_K, loss_th, ignore_index=255):
        super().__init__()
        self.K = min_K
        self.threshold = loss_th
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )

    def forward(self, logits, labels):
        labels = labels.to(torch.long)
        pixel_losses = self.criterion(logits, labels).contiguous().view(-1)

        mask = (pixel_losses > self.threshold)
        if torch.sum(mask).item() > self.K:
            pixel_losses = pixel_losses[mask]
        else:
            pixel_losses, _ = torch.topk(pixel_losses, self.K)
        return pixel_losses.mean()


# Loss for BiSeNet

#
class BalancedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index=255, weight=None):
        super(BalancedCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.per_cls_weight = weight

    def forward(self, input, target):
        # input (batch,n_classes,H,W)
        # target (batch,H,W)

        loss = F.cross_entropy(
            input, target, weight=self.per_cls_weight, ignore_index=self.ignore_index
        )
        return loss


# Recall CrossEntropy
class RecallCrossEntropy(torch.nn.Module):
    def __init__(self, n_classes=19, ignore_index=255):
        super(RecallCrossEntropy, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = (ignore_index)

    def forward(self, input, target):
        # input (batch,n_classes,H,W)
        # target (batch,H,W)

        pred = input.argmax(1)
        idex = (pred != target).view(-1)

        # calculate ground truth counts
        gt_counter = torch.ones((self.n_classes,)).cuda()
        gt_idx, gt_count = torch.unique(target, return_counts=True)

        # map ignored label to an exisiting one
        gt_count_clone = gt_count.clone()
        gt_count_clone[gt_idx == self.ignore_index] = gt_count[1]
        gt_idx[gt_idx == self.ignore_index] = 1
        gt_idx = gt_idx.to(torch.int64)
        gt_counter[gt_idx] = gt_count_clone.float()

        # calculate false negative counts
        fn_counter = torch.ones((self.n_classes)).cuda()
        fn = target.view(-1)[idex]
        fn_idx, fn_count = torch.unique(fn, return_counts=True)

        # map ignored label to an exisiting one
        fn_count_clone = fn_count.clone()
        fn_count_clone[fn_idx == self.ignore_index] = fn_count[1]
        fn_idx[fn_idx == self.ignore_index] = 1
        fn_idx = fn_idx.to(torch.int64)
        fn_counter[fn_idx] = fn_count_clone.float()

        weight = fn_counter / gt_counter

        CE = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        loss = weight[target] * CE
        return loss.mean()


class LogCoshDiceLoss(torch.nn.Module):
    def __init__(self, num_classes=19):
        super(LogCoshDiceLoss, self).__init__()
        self.loss = DiceLoss(num_classes=num_classes)

    def forward(self, output, target):
        x = self.loss(output, target)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)


# Ohem Cross Entropy


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thresh
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )
        self.NUM_OUTPUTS = 2

        self.BALANCE_WEIGHTS = [0.4, 1]

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        target = target.to(torch.long)
        if self.NUM_OUTPUTS == 1:
            score = [score]
        weights = self.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
                    (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=255, weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                                1.0166, 0.9969, 0.9754, 1.0489,
                                                0.8786, 1.0023, 0.9539, 0.9843,
                                                1.1116, 0.9037, 1.0865, 1.0955,
                                                1.0865, 1.1529, 1.0507]).cuda()
        self.criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=ignore_label
        )

    def forward(self, pred, target):
        return self.criterion(pred, target)


def unit_test(loss_type, num_classes: int, num_output=1):
    out_put = torch.rand(num_output, 4, num_classes, 512, 512)
    target = torch.rand(4, 512, 512)

    criterion = loss_type(num_classes)

    loss = criterion(out_put, target)

    print('Loss: {} '.format(loss))
