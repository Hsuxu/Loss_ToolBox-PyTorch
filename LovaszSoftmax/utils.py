import numpy as np


def IoU(pred, target):
    assert isinstance(pred, np.ndarray), 'prediction should be numpy.ndarray'
    assert isinstance(target, np.ndarray), 'prediction should be numpy.ndarray'
    eps = 1e-6
    pred = pred.flatten()
    target = target.flatten()
    inter = np.sum(pred * target).astype(np.float32)
    union = np.sum(pred).astype(np.float32) + np.sum(target).astype(np.float32) - inter
    iou = inter / (union + eps)
    return iou
