import torch
import torch.nn as nn
import torch.optim as optim

from TverskyLoss.binarytverskyloss import FocalBinaryTverskyLoss
from TverskyLoss.multitverskyloss import MultiTverskyLoss


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNet, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv3d(64, out_channels, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        out = self.Conv1(input)
        out = self.Conv2(out)
        out = self.Conv3(out)
        return out


def test():
    in_channels = 1
    n_classes = 5
    data = torch.rand(4, in_channels, 16, 64, 64)
    target = torch.randint(0, n_classes, size=(4, 1, 16, 64, 64)).long()
    net = ConvNet(in_channels, n_classes)
    opt = optim.Adam(net.parameters(), lr=0.001)
    for i in range(100):
        opt.zero_grad()
        out = net(data)
        criterion = MultiTverskyLoss(0.7, 0.3)
        loss = criterion(out, target)
        print(loss)
        loss.backward()
        opt.step()


if __name__ == '__main__':
    test()
