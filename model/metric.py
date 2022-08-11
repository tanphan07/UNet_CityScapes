import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F


def accuracy(output, target):
    # size = target.size()
    # output = F.interpolate(input=output, size=size[-2:], mode='bilinear', align_corners=True)
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == target.shape[0]
        correct = 0

        correct += torch.sum(pred == target).item()
    return correct / len(target.view(-1))


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


# def get_confusion_matrix1(output, target, number_classes):
#     cf_matrix = torch.zeros(number_classes, number_classes, dtype=torch.int64)
#     pred = torch.argmax(output, dim=1)
#     for t, p in zip(target.view(-1).type(torch.int64), pred.view(-1).type(torch.int64)):
#         cf_matrix[t, p] += 1
#
#     return cf_matrix


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=19):
    # size = mask.size()
    #
    # pred_mask = F.interpolate(input=pred_mask, size=size[-2:], mode='bilinear', align_corners=True)
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def get_confusion_matrix1(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    # output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    # print(pred.shape)
    pred = torch.permute(pred, (0, 2, 3, 1))
    # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_pred = torch.argmax(pred, dim=3).to(torch.int32)
    # print('seg_pred: {}'.format(seg_pred.shape))
    seg_gt = label[:, :size[-2], :size[-1]].to(torch.int32)
    # seg_gt = label.to(torch.int32)
    # print('seg_gt: {}'.format(seg_gt.shape))
    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    # index = (seg_gt * num_class + seg_pred).astype('int32')
    index = (seg_gt * num_class + seg_pred).to(torch.int32)
    label_count = index.bincount(minlength=num_class * num_class)
    confusion_matrix = torch.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


#

def mIoU1(pred, label, n_classes=19, ignore=255):
    size = label.size()
    cfs_matrix = get_confusion_matrix1(label, pred, size=size, num_class=n_classes, ignore=ignore)
    pos = cfs_matrix.sum(1)
    res = cfs_matrix.sum(0)
    tp = np.diag(cfs_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = (IoU_array).mean()
    return mean_IoU
