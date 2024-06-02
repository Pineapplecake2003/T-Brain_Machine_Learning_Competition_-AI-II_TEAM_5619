import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        # faltten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        Tversky_loss = (1 - Tversky)
        
        return Tversky_loss

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        batch_num = pred.shape[0]
        total_pixel = pred.shape[1] * pred.shape[2] * pred.shape[3]
        total_loss = 0.0
        pred_clamp = torch.clamp(pred, min=1e-7, max=1-1e-7)
        for i in range(batch_num):
            loss = -(1/total_pixel) * torch.sum(
                self.pos_weight * target[i, :, :, :] * torch.log(pred_clamp[i , :, :, :]) + 
                (1 - target[i , :, :, :]) * torch.log((1 - pred_clamp[i , :, :, :]))
            )
            total_loss += loss
        return total_loss / batch_num