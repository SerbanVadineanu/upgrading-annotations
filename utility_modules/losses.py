import torch.nn as nn
import torch
import torch.nn.functional as F


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=-1):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        target = target.type(torch.LongTensor)

        target = target.to(predict.get_device())
        loss = torch.nn.functional.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


class CrossEntropy2dStates(nn.Module):
    def __init__(self):
        super(CrossEntropy2dStates, self).__init__()
        self.ce = CrossEntropy2d()

    def forward(self, predict, target):
        loss = 0.0
        for pred in predict:
            loss += self.ce(pred, target)

        return loss / len(predict)


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-7, test=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.test = test

    def forward(self, predict, target):
        """Calculate Dice Loss for multiple class output.
        Args:
            predict (torch.Tensor): Model output of shape (N x C x H x W).
            target  (torch.Tensor): Target of shape (N x H x W).

        Returns:
            torch.Tensor: Loss value.
        """
        predict = nn.Softmax2d()(predict)

        if self.test:
            predict = torch.round(predict)

        target_one_hot = F.one_hot(target, num_classes=predict.shape[1]).float().permute(0, 3, 1, 2)
        mask = 1 - target_one_hot[:, 0, ...].unsqueeze(1) # We ignore label 0

        intersection = torch.sum(predict * target_one_hot * mask, dim=(2, 3))
        union = torch.sum(predict + target_one_hot - predict * target_one_hot * mask, dim=(2, 3))

        dice_coefficient = (2 * intersection) / (union + self.smooth)
        dice_loss = 1 - torch.mean(dice_coefficient)
    
        return dice_loss
    

class DiceLossSigmoid(nn.Module):
    '''
    Dice loss done only on the last output mask.
    '''
    
    def __init__(self, smooth=1e-7, test=False, ignore_label=-1):
        super(DiceLossSigmoid, self).__init__()
        self.smooth = smooth
        self.test = test
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """Calculate Dice Loss for multiple class output.
        Args:
            predict (torch.Tensor): Model output of shape (N x C x H x W).
            target  (torch.Tensor): Target of shape (N x H x W).

        Returns:
            torch.Tensor: Loss value.
        """

        loss = 0
        
        if not self.test:
            predict = torch.nn.Sigmoid()(predict)    # We are only interested in the mask corresponding to the cell
            #         predict = predict / predict.max()
        target = target.to(predict.get_device()).view(*predict.shape)

        intersection = torch.sum(predict * target * (target != self.ignore_label).float())
        union = torch.sum(predict * (target != self.ignore_label).float()) + torch.sum(target * (target != self.ignore_label).float())
        
        f1 = (2 * intersection) / (union + self.smooth)
        loss += (1 - f1)

        return loss
    
    
class DiceLoss2(nn.Module):
    '''
    Dice loss done only on the last output mask.
    '''
    
    def __init__(self, smooth=1e-7, ignore_label=-1):
        super(DiceLoss2, self).__init__()
        self.smooth = smooth
        self.ignore_label = ignore_label
        
    def forward(self, predict, target):
        """Calculate Dice Loss for multiple class output.
        Args:
            predict (torch.Tensor): Model output of shape (N x C x H x W).
            target  (torch.Tensor): Target of shape (N x H x W).

        Returns:
            torch.Tensor: Loss value.
        """

        loss = 0
        
        predict = torch.nn.Softmax2d()(predict)[:, 1, :, :]    # We are only interested in the mask corresponding to the cell
        target = target.to(predict.get_device()).view(*predict.shape)

        intersection = torch.sum(predict * target * (target != self.ignore_label).float())
        union = torch.sum(predict * (target != self.ignore_label).float()) + torch.sum(target * (target != self.ignore_label).float())
        
        f1 = (2 * intersection) / (union + self.smooth)
        loss += (1 - f1)

        return loss
    
    
class SoftDiceLoss2(nn.Module):
    '''
    Dice loss done only on the last output mask.
    '''
    
    def __init__(self, smooth=1e-7):
        super(SoftDiceLoss2, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        """Calculate Dice Loss for multiple class output.
        Args:
            predict (torch.Tensor): Model output of shape (N x C x H x W).
            target  (torch.Tensor): Target of shape (N x H x W).

        Returns:
            torch.Tensor: Loss value.
        """

        loss = 0
        
        predict = torch.nn.Softmax2d()(predict)[:, 1, :, :]    # We are only interested in the mask corresponding to the cell
        target = target.to(predict.get_device()).view(*predict.shape)

        intersection = torch.sum(predict * target)
        union = torch.sum(predict**2 + target**2)
        

        f1 = (2 * intersection + self.smooth) / (union + self.smooth)
        loss += (1 - f1)

        return loss


class DiceLossCombined(nn.Module):
    '''
    Dice loss done only on the last output mask.
    '''
    
    def __init__(self, smooth=1e-7, test=False):
        super(DiceLossCombined, self).__init__()
        self.smooth = smooth
        self.test = test
        self.dice_sig = DiceLossSigmoid()

    def forward(self, predict, target):
        """Calculate Dice Loss for multiple class output.
        Args:
            predict (torch.Tensor): Model output of shape (N x C x H x W).
            target  (torch.Tensor): Target of shape (N x H x W).

        Returns:
            torch.Tensor: Loss value.
        """

        loss_res = 0
        loss_mag = 0
        
        loss_res = self.dice_sig(predict[:, 0, :, :].unsqueeze(dim=1), target[:, 0, :, :].unsqueeze(dim=1))
        loss_mag = self.dice_sig(predict[:, 1, :, :].unsqueeze(dim=1), target[:, 1, :, :].unsqueeze(dim=1))
        
        loss = 0.8*loss_res + 0.2*loss_mag

        return loss