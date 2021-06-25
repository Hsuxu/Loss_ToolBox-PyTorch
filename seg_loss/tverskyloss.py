import torch
import torch.nn as nn

from torch.autograd import Function


class FocalBinaryTverskyLoss(Function):

    @staticmethod
    def forward(ctx, input, target):
        _alpha = 0.5
        _beta = 0.5
        _gamma = 1.0
        _epsilon = 1e-6
        _reduction = 'mean'

        batch_size = input.size(0)
        _, input_label = input.max(1)

        input_label = input_label.float()
        target_label = target.float()

        ctx.save_for_backward(input, target_label)

        input_label = input_label.view(batch_size, -1)
        target_label = target_label.view(batch_size, -1)

        ctx.P_G = torch.sum(input_label * target_label, 1)  # TP
        ctx.P_NG = torch.sum(input_label * (1 - target_label), 1)  # FP
        ctx.NP_G = torch.sum((1 - input_label) * target_label, 1)  # FN

        index = ctx.P_G / (ctx.P_G + _alpha * ctx.P_NG + _beta * ctx.NP_G + _epsilon)
        loss = torch.pow((1 - index), 1 / _gamma)
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if _reduction == 'none':
            loss = loss
        elif _reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

    # @staticmethod
    def backward(ctx, grad_out):
        """
        :param ctx:
        :param grad_out:
        :return:
        d_loss/dT_loss=(1/gamma)*(T_loss)**(1/gamma-1)
        (dT_loss/d_P1)  = 2*P_G*[G*(P_G+alpha*P_NG+beta*NP_G)-(G+alpha*NG)]/[(P_G+alpha*P_NG+beta*NP_G)**2]
                        = 2*P_G
        (dT_loss/d_p0)=
        """
        _alpha = 0.5
        _beta = 0.5
        _gamma = 1.0
        _reduction = 'mean'
        _epsilon = 1e-6

        inputs, target = ctx.saved_tensors
        inputs = inputs.float()
        target = target.float()
        batch_size = inputs.size(0)
        sum = ctx.P_G + _alpha * ctx.P_NG + _beta * ctx.NP_G + _epsilon
        P_G = ctx.P_G.view(batch_size, 1, 1, 1, 1)
        if inputs.dim() == 5:
            sum = sum.view(batch_size, 1, 1, 1, 1)
        elif inputs.dim() == 4:
            sum = sum.view(batch_size, 1, 1, 1)
            P_G = ctx.P_G.view(batch_size, 1, 1, 1)
        sub = (_alpha * (1 - target) + target) * P_G

        dL_dT = (1 / _gamma) * torch.pow((P_G / sum), (1 / _gamma - 1))
        dT_dp0 = -2 * (target / sum - sub / sum / sum)
        dL_dp0 = dL_dT * dT_dp0

        dT_dp1 = _beta * (1 - target) * P_G / sum / sum
        dL_dp1 = dL_dT * dT_dp1
        grad_input = torch.cat((dL_dp1, dL_dp0), dim=1)
        # grad_input = torch.cat((grad_out.item() * dL_dp0, dL_dp0 * grad_out.item()), dim=1)
        return grad_input, None


class BinaryTverskyLossV2(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean'):
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative. Larger beta weigh recall higher
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLossV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = 10
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target, mask=None):
        batch_size = output.size(0)
        bg_target = 1 - target
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)
            bg_target = bg_target.float().mul(valid_mask)

        output = torch.sigmoid(output).view(batch_size, -1)
        target = target.view(batch_size, -1)
        bg_target = bg_target.view(batch_size, -1)

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * bg_target, 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss


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
            loss_idx = loss_func.apply(input_idx, target_idx)
            weight_losses += loss_idx * weights[idx]
        # loss = torch.Tensor(weight_losses)
        # loss = loss.to(inputs.device)
        # loss = torch.sum(loss)
        return weight_losses
