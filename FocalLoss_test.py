import os
import torch
import unittest

import torch.nn as nn
import torch.nn.functional as F

from FocalLoss import FocalLoss

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class FocalLossTest(unittest.TestCase):
    def testFocalLoss(self):
        num_class = 5
        # alpha = np.random.randn(num_class)
        # input = torch.randn(10, num_class).cuda()
        # target = torch.LongTensor(10).random_(num_class).cuda()
        # loss0 = FL(input, target)
        # print(loss0)
        nodes = 100
        N = 100
        m = torch.nn.Linear(nodes, num_class).cuda()
        FL = FocalLoss(num_class=num_class, alpha=0.25, gamma=2.0, balance_index=2)
        for i in range(100):
            input = torch.rand(N, nodes) * torch.randint(1, 100, (N, nodes))
            input = input.cuda()

            target = torch.LongTensor(N).random_(num_class).cuda()
            loss0 = FL(m(input), target)
            print(loss0)
            loss0.backward()


if __name__ == '__main__':
    unittest.main()
