import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, logits, labels):
        logpt = self.CE_loss(logits, labels)
        pt = torch.exp(-logpt)
        loss = ((1 - pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()