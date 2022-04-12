import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

from torch.distributions.normal import Normal

class Track1ClassLoss(nn.Module):
    '''loss = bce/focal'''
    def __init__(self,loss='cross_entropy_loss',aux_loss='centre_loss'):
        super(Track1ClassLoss,self).__init__()
        if loss == 'cross_entropy_loss':
            self.loss_fn = self.cross_entropy_loss()
        elif loss == 'focal_loss': 
            self.loss_fn = self.focal_loss
    def cross_entropy_loss(self):
        loss = nn.CrossEntropyLoss()
        def lossFn(y_pred,y_true):
            return loss(y_pred,y_true)
        return lossFn

    def focal_loss(self,alpha=1.0,gamma=2.0):
        '''implemented in reduce mean form'''
        def lossFn(y_pred,y_true):
            y_pred = torch.where(y_true==1,y_pred,1-y_pred)
            loss = - 1*(1-y_pred)**self.gamma*torch.log(y_pred+1e-5)
            loss = torch.where(y_true==1,alpha*loss,loss)
            return torch.mean(loss)
        return lossFn

class Track1SegLoss(nn.Module):
    '''loss = soft_dice/soft_channel_dice'''
    def __init__(self,loss='soft_dice_loss'):
        super(Track1SegLoss,self).__init__()
        if loss == 'soft_dice_loss':
            self.loss_fn = self.soft_dice
        elif loss == 'soft_channel_dice_loss':
            self.loss_fn = self.soft_channel_dice
        assert self.loss_fn != None, f"The loss {loss} does not exist in current implementation"

    def soft_dice(self,y_pred,y_true,dim=(1,2,3),mask=None,smooth=1e-5):
        inse = torch.sum(y_true*y_pred,dim=dim)
        l = torch.sum(y_true,dim=dim)
        r = torch.sum(y_pred,dim=dim)
        dice = (2*inse + smooth)/(l+r+smooth)
        return 1-torch.mean(dice)

    def soft_channel_dice(self,y_pred,y_true,dim=(2,3),mask=None,smooth=1e-5):
        inse = torch.sum(y_true*y_pred,dim=dim)
        l = torch.sum(y_true,dim=dim)
        r = torch.sum(y_pred,dim=dim)
        dice = (2*inse + smooth)/(l+r+smooth) # channel level
        dice = torch.mean(dice,dim=1) # channel level mean
        return 1-torch.mean(dice)

    def forward(self,y_pred_,y_true,mask=None):
        return self.loss_fn(y_pred_,y_true)

class Track2Loss(nn.Module):
    '''mode = bayesian/mse/mae'''
    def __init__(self,mode='mse_loss'):
        super(Track2Loss,self).__init__()
        if mode == 'bayesian':
            self.reg_fn = self.nll_loss
        elif mode == 'mse_loss':
            self.reg_fn = self.mse()
        elif mode == 'mae_loss':
            self.reg_fn = self.mae()
        self.cross_entropy = self.cross_entropy_loss()
    def nll_loss(self,y_pred,y_true,mask):
        epsilon = 1e-5
        loss = torch.zeros_like(y_pred)
        cnt = 0
        for i in range(0,y_pred.shape[1],2):
            loss[:,cnt] = Normal(
                loc=y_pred[:,i],
                scale=torch.clamp(torch.exp(y_pred[:,i+1])+epsilon,epsilon,1e+3),
                validate_args=False
            ).log_prob(y_true[:,cnt])
            cnt += 1
        loss[:,cnt] = loss[:,cnt]*mask[:,cnt]
        return torch.mean(loss)
    def mse(self):
        mse_loss = nn.MSELoss()
        def loss(y_pred,y_true,mask):
            return mse_loss(y_pred*mask,y_true*mask)
        return loss
    def mae(self):
        mae_loss = nn.L1Loss()
        def loss(y_pred,y_true,mask):
            return mae_loss(y_pred*mask,y_true*mask)
        return loss
    def cross_entropy_loss(self):
        cross_entropy_loss_ = nn.BCELoss()
        def loss(y_pred,y_true,mask):
            return cross_entropy_loss_(y_pred,y_true*mask)
        return loss
    def forward(self,y_pred,y_true,mask):
        return {
            'regression_loss' : self.reg_fn(y_true=y_true[:,:-3],y_pred=nn.functional.relu(y_pred[:,:-3]),mask=mask[:,:-3]),
            'classification_loss' : self.cross_entropy(y_true=y_true[:,-3:],y_pred=y_pred[:,-3:],mask=mask[:,-3:])
        }





class MLDiceLoss(nn.Module):
    r"""Creates a criterion that measures and maximizes Dice Error
    between each element in the input :math:`X` and target :math:`Y`.
    Dice Cofficient between inputs :math:`X` and :math:`Y` is computed as:
    .. math:: DC(X_{c},Y_{c}) = \frac{2 \cdot | X_{c} \circ Y_{c} |}{|X_{c}| + |Y_{c}| + \epsilon}
    where :math:`\epsilon` is a constant added for numerical stability and `c` is the channel index.
    Dice Loss is computed as:
    .. math:: Loss_{DC}(X,Y) = \sum_{c} - w_{c} \cdot DC(X_{c},Y_{c})
    where,
    .. math:: w_{c} = \frac{e^{|Y_{c}|}}{\sum_{\hat{c}}e^{|Y_{\hat{c}}|}}
    Please note that Dice Loss computed finally will be negated as our
    intention is to maximize Dice Loss. General PyTorch optimizers can be
    employed to minimize Dice Loss.
    Parameters
    ----------
    eps : float
        epsilon
    References
    ----------
    https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    """

    def __init__(self, eps=1e-7):
        """Initialise loss module.
        Parameters
        ----------
        eps : float
            epsilon
        """
        super(MLDiceLoss, self).__init__()
        self.eps = eps


    def forward(self, predicted, target,dim=(2,3)):
        """Compute loss between :attr:`predicted` and :attr:`target`.
        :attr:`predicted` and :attr:`target` are tensors of shape :math:`[B,C,H,W]`
        Parameters
        ----------
        predicted : torch.Tensor
            Predicted output tensor from a model.
        target : torch.Tensor
            Ground truth tensor.
        Returns
        -------
        torch.Tensor
            MultiLabel Dice loss computed between :attr:`predicted` and :attr:`target`.
        """
        logging.debug("Inside dice loss forward routine")
        predicted = predicted.float()
        target = target.float()

        intersection = torch.sum(target * predicted,dim=dim)

        target_o = torch.sum(target, dim=dim)
        predicted_o = torch.sum(predicted, dim=dim)

        denominator = target_o + predicted_o 

        dice_loss = (2.0 * intersection) /(denominator + self.eps)
        
        w = F.softmax(1.0/(target_o+1e-3),dim=1)

        ml_dice_loss = torch.sum(dice_loss*w,dim=1)

        return -1*ml_dice_loss.mean()