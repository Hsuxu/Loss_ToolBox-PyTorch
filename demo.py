import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from utils.utils import IoU
from TverskyLoss.binarytverskyloss import FocalBinaryTverskyLoss
from DiceLoss.multi_dice_loss import MultiDiceLoss
from LovaszSoftmax.lovasz_loss import LovaszSoftmax

image_paths = ['./Data/338/t2w.mha', './Data/338/adc.mha', './Data/338/dwi.mha']
seg_path = './Data/338/mask.mha'
images = []
scale = 0.75
for image_path in image_paths:
    itkimage = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(itkimage)
    image = np.transpose(image, (1, 2, 0))
    hei, wid, _ = image.shape
    image = cv2.resize(image, (int(scale * wid), int(scale * hei)), cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    image = np.asarray(image, dtype=np.float32)
    image = (image - np.mean(image)) / np.std(image)
    images.append(image)
images = np.array(images)

itkimage = sitk.ReadImage(seg_path)
seg = sitk.GetArrayFromImage(itkimage)

seg = np.transpose(seg, (1, 2, 0))
hei, wid, _ = seg.shape
seg = cv2.resize(seg, (int(scale * wid), int(scale * hei)), cv2.INTER_NEAREST)
seg = np.transpose(seg, (2, 0, 1))
print(images.shape)
print(seg.shape)

seg = np.asarray(seg, np.int64)
data = torch.from_numpy(images)
target = torch.from_numpy(seg)
data = data.unsqueeze(0).cuda()
target = target.unsqueeze(0).cuda()


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
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


iters = 100
model = Net(len(image_paths), 3).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
dice_loss = MultiDiceLoss([0.5, 0.5, 0.5], num_class=3, k=1, ohem=True)
ious = []
losses = []
for i in range(iters):
    optimizer.zero_grad()
    out1 = model(data)
    loss, _ = dice_loss(out1, target)
    loss.backward()
    optimizer.step()
    _, pred1 = out1.max(1)
    iou = IoU(pred1.cpu().numpy(), target.cpu().numpy())
    ious.append(iou)
    losses.append(loss.item())
plt.plot(ious)
plt.plot(losses)
plt.show()
print(np.min(losses))
