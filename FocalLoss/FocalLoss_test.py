import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from focal_loss import FocalLoss_Ori, BinaryFocalLoss


# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def test_BFL():
    import matplotlib.pyplot as plt
    from torch import optim
    torch.manual_seed(123)
    shape = (4, 1, 32, 32, 32)
    datas = 40 * (torch.randint(0, 2, shape) - 0.5)
    target = torch.zeros_like(datas) + torch.randint(0, 2, size=shape)
    model = nn.Sequential(*[nn.Conv3d(1, 16, kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm3d(16),
                            nn.ReLU(),
                            nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1)])

    criterion = BinaryFocalLoss()
    losses = []
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    for step in range(100):
        out = model(datas)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(step)

    plt.plot(losses)
    plt.show()


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
    FL = FocalLoss_Ori(num_class=num_class, alpha=0.25,
                       gamma=2.0, balance_index=2)
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
        output = F.softmax(output, dim=1)
        loss = FL(output, target)
        print(loss.item())


if __name__ == '__main__':
    test_BFL()
