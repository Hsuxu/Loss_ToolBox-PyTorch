import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes=None):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Shapes:
        predict: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with predict
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    if num_classes is None:
        num_classes = input.max() + 1
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        output = output.contiguous().view(output.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.pow(2) + target.pow(2), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        output: A tensor of shape [N, C, *]
        target: A tensor of same shape with output
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=[], **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif ignore_index is None:
            self.ignore_index = []
        self.ignore_index = ignore_index

    def forward(self, output, target):
        assert output.shape == target.shape, 'output & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        output = F.softmax(output, dim=1)
        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = dice(output[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += (dice_loss)
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss


class BCE_DiceLoss(nn.Module):
    def __init__(self, alpha=0.5, ignore_index=None, reduction='mean'):
        """
        combination of Binary Cross Entropy and Binary Dice Loss
        Args:
            @param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            @param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            @param alpha: weight between  BCE('Binary Cross Entropy') and binary dice
        Shapes:
            output: A tensor of shape [N, *] without sigmoid activation function applied
            target: A tensor of shape same with output
        """
        super(BCE_DiceLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        assert (alpha >= 0 and alpha <= 1), '`alpha` should in [0,1]'
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dice = BinaryDiceLoss(ignore_index=ignore_index, reduction=reduction)
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, output, target):
        dice_loss = self.dice(output, target)

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            output = output.mul(mask)  # can not use inplace for bp
            target = target.float().mul(mask)
        bce_loss = self.bce(output, target)
        loss = self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


def test():
    input = torch.rand((1, 1, 32, 32, 32))
    model = nn.Conv3d(1, 1, 3, padding=1)
    target = torch.randint(0, 3, (1, 1, 32, 32, 32)).float()
    criterion = BCE_DiceLoss(ignore_index=2, reduction='none')
    loss = criterion(model(input), target)
    loss.backward()
    print(loss.item())

    # input = torch.zeros((1, 2, 32, 32, 32))
    # input[:, 0, ...] = 1
    # target = torch.ones((1, 1, 32, 32, 32)).long()
    # target_one_hot = make_one_hot(target, num_classes=2)
    # # print(target_one_hot.size())
    # criterion = DiceLoss()
    # loss = criterion(input, target_one_hot)
    # print(loss.item())


if __name__ == '__main__':
    test()
