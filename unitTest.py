import os
import torch
import unittest

import torch.nn as nn
import torch.nn.functional as F

from FocalLoss import FocalLoss

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class FocalLossTest(unittest.TestCase):

    def testFocalLoss(self):
        focal = FocalLoss(alpha=[1, 1], gamma=2)
        # CE = FL(class_num=2, alpha=[1, 1],
        #         gamma=2)
        CE = nn.NLLLoss()
        maxe = 0
        for i in range(100):
            data = torch.rand(4096, 2).cuda() * torch.randint(1, 100, (1,)).cuda()
            label = torch.rand(4096).ge(0.1).long().cuda()
            loss0 = focal(data, label)
            loss1 = CE(data, label)
            if abs(loss1.item() - loss0.item()) > maxe:
                maxe = abs(loss1.item() - loss0.item())
        print('error: {}'.format(maxe))


if __name__ == '__main__':
    unittest.main()
