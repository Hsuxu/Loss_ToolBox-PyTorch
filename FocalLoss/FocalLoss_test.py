import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from FocalLoss import FocalLoss

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def test_focal():
    num_class = 5
    # alpha = np.random.randn(num_class)
    # input = torch.randn(10, num_class).cuda()
    # target = torch.LongTensor(10).random_(num_class).cuda()
    # loss0 = FL(input, target)
    # print(loss0)
    nodes = 100
    N = 100
    # model1d = torch.nn.Linear(nodes, num_class).cuda()
    model2d = torch.nn.Conv2d(16, num_class, 3, padding=1).cuda()
    FL = FocalLoss(num_class=num_class, alpha=0.25, gamma=2.0, balance_index=2)
    for i in range(10):
        # input = torch.rand(N, nodes) * torch.randint(1, 100, (N, nodes)).float()
        # input = input.cuda()
        # target = torch.LongTensor(N).random_(num_class).cuda()
        # loss0 = FL(model1d(input), target)
        # print(loss0)
        # loss0.backward()

        input = torch.rand(3, 16, 32, 32).cuda()
        target = torch.rand(3, 32, 32).random_(num_class).cuda()
        target = target.long().cuda()
        output = model2d(input)
        output = F.softmax(output,dim=1)
        loss = FL(output, target)
        print(loss.item())


if __name__ == '__main__':
    test_focal()