import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = F.softmax(inputs, dim=1)

        smooth = 1.0
        loss = 0.0

        for c in range(self.num_classes):
            input_c = inputs[:, c, :, :]
            target_c = (target == c).float()

            input_flat = input_c.contiguous().view(-1)
            target_flat = target_c.contiguous().view(-1)

            intersection = (input_flat * target_flat).sum()
            denominator = input_flat.sum() + target_flat.sum() + smooth
            
            if denominator.item() == 0:
                continue

            loss += 1 - ((2. * intersection + smooth) / denominator)

        return loss / self.num_classes


def combined_loss(output, target, weight_tensor, num_classes):
    criterion_ce = nn.CrossEntropyLoss(weight=weight_tensor)
    criterion_dice = DiceLoss(num_classes)
    
    ce_loss = criterion_ce(output, target.long())
    dice_loss = criterion_dice(output, target.float(), softmax=True)
    
    return ce_loss + dice_loss
