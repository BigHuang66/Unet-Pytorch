# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Implements the cross entropy loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        # if len(label.shape) != len(logit.shape):
        #     label = torch.unsqueeze(label, 1)

        # logit = torch.transpose(logit, 1, 3)
        # label = torch.transpose(label, 1, 3)
        loss = F.cross_entropy(
            logit, label.long(), ignore_index=self.ignore_index, reduction='mean')

        # mask = label != self.ignore_index
        # mask = mask.float()
        # loss = loss * mask
        # avg_loss = torch.mean(loss) / (torch.mean(mask) + self.EPS)

        return loss
