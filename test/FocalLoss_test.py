import torch

import torch.nn as nn
from torch import optim

from seg_loss.focal_loss import FocalLoss_Ori, BinaryFocalLoss


def test_BFL():
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt
    torch.manual_seed(123)
    num_class = 5
    shape = (4, 1, 32, 32, 32)
    target_shape = (4, 1, 32, 32, 32)
    datas = 40 * (torch.rand(shape) - 0.5).cuda()
    target = torch.randint(0, num_class, size=target_shape).cuda()
    target[0, 0, 0, 0, :] = 255
    target = target.long().cuda()
    FL = FocalLoss_Ori(num_class=num_class, gamma=2.0, ignore_index=255, reduction='mean')

    model = nn.Sequential(*[nn.Conv3d(1, 16, kernel_size=3, padding=1, stride=1),
                            nn.BatchNorm3d(16),
                            nn.ReLU(),
                            nn.Conv3d(16, num_class, kernel_size=3, padding=1, stride=1)])
    model = model.cuda()
    losses = []
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    for i in range(100):
        output = model(datas)
        loss = FL(output, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(i)

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    test_focal()
