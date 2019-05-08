import torch
import torch.nn as nn
from .binarytverskyloss import FocalBinaryTverskyLoss


class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, weights=None):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`

        """
        super(MultiTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights

    def forward(self, inputs, targets):

        num_class = inputs.size(1)
        weight_losses = 0.0
        if self.weights is not None:
            assert len(self.weights) == num_class, 'number of classes should be equal to length of weights '
            weights = self.weights
        else:
            weights = [1.0 / num_class] * num_class
        input_slices = torch.split(inputs, [1] * num_class, dim=1)
        for idx in range(num_class):
            input_idx = input_slices[idx]
            input_idx = torch.cat((1 - input_idx, input_idx), dim=1)
            target_idx = (targets == idx) * 1
            loss_func = FocalBinaryTverskyLoss(self.alpha, self.beta, self.gamma)
            loss_idx = loss_func(input_idx, target_idx)
            weight_losses+=loss_idx * weights[idx]
        # loss = torch.Tensor(weight_losses)
        # loss = loss.to(inputs.device)
        # loss = torch.sum(loss)
        return weight_losses
