import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, alpha=ALPHA, gamma=GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = inputs.sigmoid()
        inputs = torch.clamp(inputs, min=1e-4, max=1.0 - 1e-4)
        # Flatten tensors
        inputs = inputs.view(inputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = inputs.sigmoid()
        inputs = torch.clamp(inputs, min=1e-4, max=1.0 - 1e-4)
        # Flatten tensors
        inputs = inputs.view(inputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        intersection = (inputs * targets).sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice_score.mean()
