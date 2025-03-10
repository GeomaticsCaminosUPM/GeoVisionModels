#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Nov 17 12:48:36 2021

@author: Nacriema

Refs:

I build the collection of loss that used in Segmentation Task, beside the Standard Loss provided by Pytorch, I also
implemented some loss that can be used to enhanced the training process.

For me: Loss function is computed by comparing between probabilities, so in each Loss function if we pass logit as input
then we should convert them into probability. One-hot encoding also a form of probability.

For testing purpose, we should crete ideal probability for compare them. Then I give the loss function option use soft
max or not.

May be I need to convert each function inside the forward pass to the function that take the input and target as softmax
probability, inside the forward pass we just convert the logits into it


Should use each function, because most other functions like Exponential Logarithmic Loss use the result of the defined
function above for compute.

Difference between BCELoss and CrossEntropy Loss when consider with mutiple classification (n_classes >= 3):
    - When I'm reading about the fomular of CrossEntropy Loss for multiple class case, then I see the loss just "inclue" the t*ln(p) part, but not the (1 - t)ln(1 - p)
    for the "background" class. Then it can not "capture" the properties between each class with the background, just between each class together. 
    - Then I'm reading from this thread https://github.com/ultralytics/yolov5/issues/5401, the author give me the same idea. 


Reference papers: 
    * https://arxiv.org/pdf/2006.14822.pdf

"""
#from __future__ import annotations

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod, ABC
from einops import rearrange
from torch import nn

from torch import nn

from typing import Optional

import torch
from torch import Tensor, nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

# based on:
# https://github.com/bermanmaxim/LovaszSoftmax


_EPSILON = 1e-8


class ComposeLoss(nn.Module):
    def __init__(self, loss_list:list, importance:list = [], predicate:str = 'sum') -> None:
        super().__init__()
        import torch
        if len(loss_list) < 2:
            raise Exception("Give at least two loss functions in loss_list")
        
        self.loss_list = loss_list 
        if len(importance) < len(loss_list):
            for i in range(len(loss_list) - len(importance)):
                importance.append(1)

        self.importance = torch.tensor(importance).float()

        for i in range(len(self.loss_list)):
            try:
                self.loss_list[i].weight = torch.tensor(self.loss_list[i].weight).float()
            except:
                try:
                    self.loss_list[i].weights = torch.tensor(self.loss_list[i].weights).float()
                except:
                    try:
                        self.loss_list[i].class_weights = torch.tensor(self.loss_list[i].class_weights).float()
                    except:
                        None

        self.predicate = predicate

    def to_device(self,device) -> None:
        #self.importance = self.importance.to_device(device)
        for i in range(len(self.loss_list)):
            try:
                self.loss_list[i].weight = self.loss_list[i].weight.to(device=device)
            except:
                try:
                    self.loss_list[i].weights = self.loss_list[i].weights.to(device=device)
                except:
                    try:
                        self.loss_list[i].class_weights = self.loss_list[i].class_weights.to(device=device)
                    except:
                        try: 
                            self.loss_list[i].to_device(device)
                        except:
                            None

    def forward(self, pred, target):
        loss_value = self.loss_list[0](pred,target) * self.importance[0]
        if self.predicate == 'prod':
            for i in range(1,len(self.loss_list)):
                loss_value *= self.loss_list[i](pred,target) * self.importance[i]

        elif self.predicate == 'sum':
            for i in range(1,len(self.loss_list)):
                loss_value += self.loss_list[i](pred,target) * self.importance[i]

        else:
            raise Exception(f"predicate {self.predicate} not implemented")
        
        return loss_value
    
def get_loss(name):
    if name is None:
        name = 'bce_logit'
    return {
        'bce': BCELoss,
        'bce_logit': BCEWithLogitsLoss,
        'cross_entropy': CrossEntropyLoss,
        'soft_dice': SoftDiceLoss,
        'bach_soft_dice': BatchSoftDice,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'log_cosh_dice': LogCoshDiceLoss,
        'sensitivity_specificity': SensitivitySpecificityLoss, 
        'exponential_logarithmic': ExponentialLogarithmicLoss,
        'combo': ComboLoss,
    }[name]


def soft_dice_loss(output, target, epsilon=1e-6):
    numerator = 2. * torch.sum(output * target, dim=(-2, -1))
    denominator = torch.sum(output + target, dim=(-2, -1))
    return (numerator + epsilon) / (denominator + epsilon)
    # return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))

# DONE
class SoftDiceLoss(nn.Module):
    def __init__(self, reduction='none', use_softmax=True):
        """
        Args:
            use_softmax: Set it to False when use the function for testing purpose
        """
        super(SoftDiceLoss, self).__init__()
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, output, target, epsilon=1e-6):
        """
        References:
        JeremyJordan's Implementation
        https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py

        Paper related to this function:
        Formula for binary segmentation case - A survey of loss functions for semantic segmentation
        https://arxiv.org/pdf/2006.14822.pdf

        Formula for multiclass segmentation cases - Segmentation of Head and Neck Organs at Risk Using CNN with Batch
        Dice Loss
        https://arxiv.org/pdf/1812.02427.pdf

        Args:
            output: Tensor shape (N, N_Class, H, W), torch.float
            target: Tensor shape (N, H, W)
            epsilon: Use this term to avoid undefined edge case

        Returns:

        """
        num_classes = output.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == one_hot_target.shape
        if self.reduction == 'none':
            return 1.0 - soft_dice_loss(output, one_hot_target)
        elif self.reduction == 'mean':
            return 1.0 - torch.mean(soft_dice_loss(output, one_hot_target))
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


# NOT SURE
class BatchSoftDice(nn.Module):
    def __init__(self, use_square=False):
        """
        Args:
            use_square: If use square then the denominator will the sum of square
        """
        super(BatchSoftDice, self).__init__()
        self._use_square = use_square

    def forward(self, output, target, epsilon=1e-6):
        """
        This is the variance of SoftDiceLoss, it in introduced in:
        https://arxiv.org/pdf/1812.02427.pdf
        Args:
            output: Tensor shape (N, N_Class, H, W), torch.float
            target: Tensor shape (N, H, W)
            epsilon: Use this term to avoid undefined edge case
        Returns:
        """
        num_classes = output.shape[1]
        batch_size = output.shape[0]
        axes = (-2, -1)
        output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2))
        assert output.shape == one_hot_target.shape
        numerator = 2. * torch.sum(output * one_hot_target, dim=axes)
        if self._use_square:
            denominator = torch.sum(torch.square(output) + torch.square(one_hot_target), dim=axes)
        else:
            denominator = torch.sum(output + one_hot_target, dim=axes)
        return (1 - torch.mean((numerator + epsilon) / (denominator + epsilon))) * batch_size
        # return 1 - torch.sum(torch.mean(((numerator + epsilon) / (denominator + epsilon)), dim=1))


# DONE
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='none', eps=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, output, target):
        num_classes = output.shape[1]
        output_softmax = F.softmax(output, dim=1)
        output_log_softmax = F.log_softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        weight = torch.pow(1.0 - output_softmax, self.gamma)
        focal = -self.alpha * weight * output_log_softmax
        # This line is very useful, must learn einsum, bellow line equivalent to the commented line
        # loss_tmp = torch.sum(focal.to(torch.float) * one_hot_target.to(torch.float), dim=1)
        loss_tmp = torch.einsum('bc..., bc...->b...', one_hot_target, focal)
        if self.reduction == 'none':
            return loss_tmp
        elif self.reduction == 'mean':
            return torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            return torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


# DONE
class TverskyLoss(nn.Module):
    """
    Tversky Loss is the generalization of Dice Loss
    It in the group of Region-Base Loss
    """
    def __init__(self, beta=0.5, use_softmax=True):
        """
        Args:
            beta:
            use_softmax: Set to False is used for testing purpose, when training model, use default True instead
        """
        super(TverskyLoss, self).__init__()
        self.beta = beta
        self.use_softmax = use_softmax

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        if self.use_softmax:
            output = F.softmax(output, dim=1)  # predicted value
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == target.shape
        # Notice: TverskyIndex is numerator / denominator
        # See https://en.wikipedia.org/wiki/Tversky_index and we have the quick comparison between probability and set \
        # G is the Global Set, A_ = G - A, then
        # |A - B| = |A ^ B_| = |A ^ (G - B)| so |A - B| in set become (1 - target) * (output)
        # With ^ = *, G = 1
        numerator = torch.sum(output * target, dim=(-2, -1))
        denominator = numerator + self.beta * torch.sum((1 - target) * output, dim=(-2, -1)) + (1 - self.beta) * torch.sum(target * (1 - output), dim=(-2, -1))
        return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))


# DONE
class FocalTverskyLoss(nn.Module):
    """
    More information about this loss, see: https://arxiv.org/pdf/1810.07842.pdf
    This loss is similar to Tversky Loss, but with a small adjustment
    With input shape (batch, n_classes, h, w) then TI has shape [batch, n_classes]
    In their paper TI_c is the tensor w.r.t to n_classes index

    FTL = Sum_index_c(1 - TI_c)^gamma
    """
    def __init__(self, gamma=1, beta=0.5, use_softmax=True):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.use_softmax = use_softmax

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        if self.use_softmax:
            output = F.softmax(output, dim=1)  # predicted value
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == target.shape
        numerator = torch.sum(output * target, dim=(-2, -1))
        denominator = numerator + self.beta * torch.sum((1 - target) * output, dim=(-2, -1)) + (
                    1 - self.beta) * torch.sum(target * (1 - output), dim=(-2, -1))
        TI = torch.mean((numerator + epsilon) / (denominator + epsilon), dim=0)  # Shape [batch, num_classes], should reduce along batch dim
        return torch.sum(torch.pow(1.0 - TI, self.gamma))


# DONE
class LogCoshDiceLoss(nn.Module):
    """
    L_{lc-dce} = log(cosh(DiceLoss)
    """
    def __init__(self, use_softmax=True):
        super(LogCoshDiceLoss, self).__init__()
        self.use_softmax = use_softmax

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == one_hot_target.shape
        numerator = 2. * torch.sum(output * one_hot_target, dim=(-2, -1))  # Shape [batch, n_classes]
        denominator = torch.sum(output + one_hot_target, dim=(-2, -1))
        return torch.log(torch.cosh(1 - torch.mean((numerator + epsilon) / (denominator + epsilon))))


# Helper function for sensitivity-specificity loss
def sensitivity_specificity_loss(y_true, y_pred, w):
    """
    True positive example (True - Reality, Positive - Wolf):
    A sentence to describe it - we make the positive prediction and this is True in Reality .
    * Reality: A wolf threatened
    * Shepherd said: "Wolf"
    * Outcome: Shepherd is a hero
    Args:
        y_true: probability (one hot) shape [batch, n_classes, h, w]
        y_pred: probability (softmax(output) or sth like that) shape [batch, n_classes, h, w]
    Returns:
        Loss: A tensor
    """
    assert y_true.shape == y_pred.shape
    n_classes = y_true.shape[1]
    confusion_matrix = torch.zeros((n_classes, n_classes), dtype=torch.float)
    y_true = torch.argmax(y_true, dim=1)  # Reduce to [batch, h, w]
    y_pred = torch.argmax(y_pred, dim=1)
    # Use trick to compute the confusion matrix
    # Reference: https://github.com/monniert/docExtractor/
    for y_true_item, y_pred_item in zip(y_true, y_pred):
        y_true_item = y_true_item.flatten()  # Reduce to 1-D tensor
        y_pred_item = y_pred_item.flatten()
        confusion_matrix += torch.bincount(n_classes * y_true_item + y_pred_item, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    # From confusion matrix, we compute tp, fp, fn, tn
    # Get the answer from this discussion:
    # https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier
    sum_along_classified = torch.sum(confusion_matrix, dim=1)  # sum(c1_1, cn_1) return 1D tensor
    sum_along_actual = torch.sum(confusion_matrix, dim=0)  # sum(c1_1 -> c1_n)
    tp = torch.diagonal(confusion_matrix, offset=0)
    fp = sum_along_classified - tp
    fn = sum_along_actual - tp
    tn = torch.ones(n_classes, dtype=torch.float) * torch.sum(confusion_matrix) - tp - fp - fn
    smooth = torch.ones(n_classes, dtype=torch.float)  # Use to avoid numeric division error
    assert tp.shape == fp.shape == fn.shape == tn.shape
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    # Relation between tp, fp, fn, tn annotation vs set annotation here, so the actual loss become, compare this
    # loss vs the Soft Dice Loss, see https://arxiv.org/pdf/1803.11078.pdf
    return 1.0 - torch.mean(w * sensitivity + (1 - w) * specificity)


# XXX Bugs
class SensitivitySpecificityLoss(nn.Module):
    def __init__(self, weight=0.5):
        """
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        Args:
            weight: use for the combination of sensitivity and specificity
        """
        super(SensitivitySpecificityLoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        num_classes = output.shape[1]
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        output = F.softmax(output, dim=1)
        return sensitivity_specificity_loss(target, output, self.weight)


# TODO: NOT IMPLEMENTED
class CompoundedLoss(nn.Module):
    def __init__(self):
        super(CompoundedLoss, self).__init__()
        pass

    def forward(self, output, target):
        pass


# DONE
class ComboLoss(nn.Module):
    """
    It is defined as a weighted sum of Dice loss and a modified cross entropy. It attempts to leverage the 
    flexibility of Dice loss of class imbalance and at same time use cross-entropy for curve smoothing. 
    
    This loss will look like "batch bce-loss" when we consider all pixels flattened are predicted as correct or not

    Paper: https://arxiv.org/pdf/1805.02798.pdf. See the original paper at formula (3)
    Author's implementation in Keras : https://github.com/asgsaeid/ComboLoss/blob/master/combo_loss.py

    This loss is perfect loss when the training loss come to -0.5 (with the default config)
    """
    def __init__(self, use_softmax=True, ce_w=0.5, ce_d_w=0.5, eps=1e-12):
        super(ComboLoss, self).__init__()
        self.use_softmax = use_softmax
        self.ce_w = ce_w
        self.ce_d_w = ce_d_w
        self.eps = 1e-12
        self.smooth = 1

    def forward(self, output, target):
        num_classes = output.shape[1]
        
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        
        # At this time, the output and one_hot_target have the same shape        
        y_true_f = torch.flatten(one_hot_target)
        y_pred_f = torch.flatten(output)
        intersection = torch.sum(y_true_f * y_pred_f)
        d = (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

        # From this thread: https://discuss.pytorch.org/t/bceloss-how-log-compute-log-0/11390. Use this trick to advoid nan when log(0) and log(1)
        out = - (self.ce_w * y_true_f * torch.log(y_pred_f + self.eps) + (1 - self.ce_w) * (1.0 - y_true_f) * torch.log(1.0 - y_pred_f + self.eps))
        weighted_ce = torch.mean(out, axis=-1)

        # Due to this is the hibird loss, then the loss can become negative: https://discuss.pytorch.org/t/negative-value-in-my-loss-function/101776
        combo = (self.ce_d_w * weighted_ce) - ((1 - self.ce_d_w) * d)
        return combo


# DONE
class ExponentialLogarithmicLoss(nn.Module):
    """
    This loss is focuses on less accurately predicted structures using the combination of Dice Loss ans Cross Entropy
    Loss
    
    Original paper: https://arxiv.org/pdf/1809.00076.pdf
    
    See the paper at 2.2 w_l = ((Sum k f_k) / f_l) ** 0.5 is the label weight
    
    Note: 
        - Input for CrossEntropyLoss is the logits - Raw output from the model
    """
    
    def __init__(self, w_dice=0.5, w_cross=0.5, gamma=0.3, use_softmax=True, class_weights=None):
        super(ExponentialLogarithmicLoss, self).__init__()
        self.w_dice = w_dice
        self.gamma = gamma
        self.w_cross = w_cross
        self.use_softmax = use_softmax
        self.class_weights = class_weights

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        assert len(self.class_weights) == num_classes, "Class weight must be not None and must be a Tensor of size C - Number of classes"
        
        # Generate the class weights array. Shape (batch_size, height, width), at pixel n, the nuber is the weight of the true class
        weight_map = self.class_weights[target]
        
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        
        l_dice = torch.mean(torch.pow(-torch.log(soft_dice_loss(output, target)), self.gamma))   # mean w.r.t to label
        l_cross = torch.mean(torch.mul(weight_map, torch.pow(F.cross_entropy(output, target, reduction='none'), self.gamma)))
        return self.w_dice * l_dice + self.w_cross * l_cross


# This is use for testing purpose
if __name__ == '__main__':
    # loss = BCEWithLogitsLoss(reduction="none")
    # loss = BCELoss()
    # loss = CrossEntropyLoss()
    # loss = FocalLoss(alpha=1.0, reduction='mean', gamma=1)
    # loss = SoftDiceLoss(reduction='mean', use_softmax=False)
    # loss = SensitivitySpecificityLoss(weight=0.5)
    # loss = LogCoshDiceLoss(use_softmax=True)
    # loss = BatchSoftDice(use_square=False)
    # loss = TverskyLoss()
    # loss = FocalTverskyLoss(use_softmax=True)
    # loss = ExponentialLogarithmicLoss(use_softmax=True, class_weights=torch.tensor([0.2, 0.4, 0.1, 0.1, 0.1, 0.1]))
    loss = ComboLoss(use_softmax=True, ce_d_w=0.5, ce_w=0.5)

    ###### Binary classification test ######
    # output = torch.randn((1, 2, 1, 1), requires_grad=True)
    # target = torch.empty((1, 1, 1), dtype=torch.float).random_(2)
    # output_ = F.one_hot(target.to(torch.int64), num_classes=2).permute((0, 3, 1, 2))
    
    ###### Multiple classes classification test ######
    batch_size = 2
    n_classes = 6
    height = 3 
    width = 5
    
    output = torch.randn((batch_size, n_classes, height, width), requires_grad=True)  # Shape: n_samples, n_classes, h, w 
    target = torch.empty((batch_size, height, width), dtype=torch.long).random_(n_classes)   # Shape: n_samples, h, w, each cell represent the class index
    output_ = F.one_hot(target.to(torch.int64), num_classes=n_classes).permute((0, 3, 1, 2)).to(torch.float)  # Mimic the "ideal" model output after going through sigmoid function
    output_.requires_grad = True
    
    print(f'Output shape: {output.shape}')
    print(f'Output_ shape: {output_.shape}')
    print(f'Target shape: {target.shape}')

    # TEST: Test loss between the logit output of the model and the groud truth label then we need to enable the use_softmax=True flag to True when init the loss function to test this
    loss_1 = loss(output, target) 
    print(f'Loss 1 value: {loss_1}')
    loss_1.backward()
    print(output.grad)  
    
    # TEST: Test loss function when the input and target are the same (model prediction will be output like the one-hot encoded vector, so set the use_softmax=False when init the loss function to test this)
    # loss_2 = loss(output_, target)   
    # print(f'Loss 2 value: {loss_2}')
    # loss_2.backward()
    # print(output.grad)


def _tversky_index_c(
    p: torch.Tensor,
    g: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = _EPSILON,
    reduction="sum",
    #class_weights=None,
) -> torch.Tensor:
    """Compute the Tversky similarity index for each class for predictions p and
    ground truth labels g.

    Args:
        p : np.ndarray shape=(batch_size, num_classes, height, width)
            Softmax or sigmoid is applied afterwards.
        g : np.ndarray shape=(batch_size, height, width)
            int type ground truth labels for each sample.
        alpha : Optional[float]
            The relative weight to go to false negatives.
        beta : Optional[float]
            The relative weight to go to false positives.
        smooth : Optional[float]
            A function smooth parameter that also provides numerical stability.
        reduction: Optional[str]
            The reduction method to apply to the output. Must be either 'sum' or 'none'.

    Returns:
        List[float]
            The calculated similarity index amount for each class.
    """
    if p.shape[1] > 1:
        p = torch.nn.functional.softmax(p, dim=1)
    else:
        p = torch.nn.functional.sigmoid(p, dim=1)

    tp = torch.mul(p, g)
    fn = torch.mul(1.0 - p, g)
    fp = torch.mul(p, 1.0 - g)

    #if type(class_weights) == type(None):
    if reduction == "sum":
        tp = torch.nansum(tp, dim=0)
        fn = torch.nansum(fn, dim=0)
        fp = torch.nansum(fp, dim=0)
    elif reduction != "none":
        raise ValueError("Reduction must be either 'sum' or 'none'.")
    
    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    #else:
    #    loss = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    #    # class_weights = class_weights / (torch.sum(class_weights) * class_weights.shape[0])
    #    loss = torch.einsum('j,ij->ij', class_weights, loss)
    #    if reduction == "sum":
    #        loss = torch.nansum(loss, dim=0)
    #    elif reduction != "none":
    #        raise ValueError("Reduction must be either 'sum' or 'none'.")
    #    
    #    return loss



def _dice_similarity_c(
    p: torch.Tensor, g: torch.Tensor, smooth: float = _EPSILON, reduction="sum",#class_weights=None,
) -> torch.Tensor:
    """Compute the Dice similarity index for each class for predictions p and ground
    truth labels g.

    Args:
        p : np.ndarray shape=(batch_size, num_classes, height, width)
            Predictions. Softmax or sigmoid scaled is applied apferwards.
        g : np.ndarray shape=(batch_size, height, width)
            int type ground truth labels for each sample.
        smooth : Optional[float]
            A function smooth parameter that also provides numerical stability.
        reduction: Optional[str]
            The reduction method to apply to the output. Must be either 'sum' or 'none'.

    Returns:
        List[float]
            The calculated similarity index amount for each class.
    """
    return _tversky_index_c(
        p, g, alpha=0.5, beta=0.5, smooth=smooth, reduction=reduction,# class_weights=class_weights
    )


class _Loss(nn.Module, ABC):
    ignore_index = None

    @abstractmethod
    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass.")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_weights = None) -> torch.Tensor:
        """Calculate loss.

        Args:
            y_pred : Tensor of shape (batch_size, num_classes, ...).
                Predicted probabilities for each output class. Softmax or sigmoid is applied afterwards if needed. 
            y_true : Tensor of shape (batch_size, ...)
                Ground truth labels.

        Returns:
            loss : Loss value.
        """
        """Calculate loss.

        Args:
            y_pred : Tensor of shape (batch_size, num_classes, ...).
                Predicted probabilities for each output class. Softmax or sigmoid is applied afterwards if needed.
            y_true : Tensor of shape (batch_size, ...)
                Ground truth labels.

        Returns:
            loss : Loss value.
        """

        # Flatten tensors
        y_true = rearrange(y_true, "n ... -> (n ...)")
        y_pred = rearrange(y_pred, "n c ... -> (n ...) c")
        n, c = y_pred.shape

        # Remove ignore class
        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        # One-hot encode y_true
        y_true = F.one_hot(y_true, num_classes=c)

        # Calculate loss
        if type(class_weights) == type(None):
            return self._calc_loss(y_pred, y_true)
        else:
            return self._calc_loss(y_pred, y_true,class_weights=class_weights)


################################
#       Dice coefficient       #
################################
class DiceCoefficient(_Loss):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply
        Dice coefficient, is a statistical tool which measures the similarity between
        two sets of data.

    Args:
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7.
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self, delta: float = 0.7, smooth: float = _EPSILON, ignore_index: int = None,# class_weights=None
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.ignore_index = ignore_index
        #self.class_weights = class_weights

    #def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_weights=None) -> torch.Tensor:
    #    if type(class_weights) == type(None):
    #        class_weights = self.class_weights
    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_class = _dice_similarity_c(y_pred, y_true, smooth=self.smooth)#, class_weights=class_weights)
        return torch.mean(dice_class)


################################
#           Dice loss          #
################################
class DiceLoss(_Loss):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic
    developed in the 1940s to gauge the similarity between two samples.

    Args:
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7.
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self, delta: float = 0.7, smooth: float = _EPSILON, ignore_index: int = None,# class_weights=None
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.ignore_index = ignore_index
        #self.class_weights = class_weights

    #def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor,class_weights=None) -> torch.Tensor:
    #    if type(class_weights) == type(None):
    #        class_weights = self.class_weights
    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_class = _dice_similarity_c(y_pred, y_true, smooth=self.smooth)#, class_weights = class_weights)
        return torch.mean(1 - dice_class)


################################
#         Tversky loss         #
################################
class TverskyLoss(_Loss):
    """Tversky loss function for image segmentation using 3D fully convolutional deep
    networks. Link: https://arxiv.org/abs/1706.05721

    Args:
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7.
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self, delta: float = 0.7, smooth: float = _EPSILON, ignore_index: int = None,# class_weights=None
    ):
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.ignore_index = ignore_index
        #self.class_weights = class_weights

    #def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor,class_weights=None) -> torch.Tensor:
    #    if type(class_weights) == type(None):
    #        class_weights = self.class_weights
    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        tversky_class = _tversky_index_c(
            y_pred, y_true, alpha=self.delta, beta=1 - self.delta, smooth=self.smooth, #class_weights=class_weights
        )
        return torch.mean(1 - tversky_class)


################################
#      Focal Tversky loss      #
################################
class FocalTverskyLoss(_Loss):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion
        segmentation
    Link: https://arxiv.org/abs/1810.07842

    Args:
        delta : float, optional
            controls weight given to each class, by default 0.7
        gamma : float, optional
            focal parameter controls degree of down-weighting of easy examples,
            by default 0.75
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = _EPSILON,
        ignore_index: int = None,
        #class_weights = None
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index
        #self.class_weights = class_weights

    #def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor,class_weights=None) -> torch.Tensor:
    #    if type(class_weights) == type(None):
    #        class_weights = self.class_weights
    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        tversky_class = _tversky_index_c(
            y_pred, y_true, alpha=self.delta, beta=1 - self.delta, smooth=self.smooth#, class_weights=class_weights
        )
        # Average class scores
        return torch.mean(torch.pow((1 - tversky_class), self.gamma))


################################
#          Focal loss          #
################################
class FocalLoss(_Loss):
    """Focal loss is used to address the issue of the class imbalance problem.
        A modulation term applied to the Cross-Entropy loss function.

    Args:
        delta : float, optional
            controls relative weight of false positives and false negatives. delta > 0.5
            penalises false negatives more than false positives, by default 0.7.
        gamma : float, optional
            focal parameter controls degree of down-weighting of easy examples,
            by default 0.75.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self, delta: float = 0.7, gamma: float = 0.75, ignore_index: int = None, class_weights=None
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_weights=None) -> torch.Tensor:
        if type(class_weights) == type(None):
            class_weights = self.class_weights
    
        if y_pred.shape[1] > 1:
            y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
        else:
            y_pred = torch.nn.functional.log_sigmoid(y_pred, dim=1)

        cross_entropy = -y_true * y_pred#torch.log(y_pred + _EPSILON)

        if self.delta is not None:
            focal_loss = (
                self.delta
                * torch.pow(1 - y_pred.exp(), self.gamma)
                * cross_entropy
            )
        else:
            focal_loss = torch.pow(1 - y_pred.exp(), self.gamma) * cross_entropy

        if type(class_weights) != type(None):
            #class_weights = class_weights / (torch.sum(class_weights) * class_weights.shape[0])
            focal_loss = torch.einsum('j,ij->ij', class_weights, focal_loss)

        return torch.mean(torch.sum(focal_loss, dim=1))


################################
#          Combo loss          #
################################
class ComboLoss(_Loss):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798

    Args:
        alpha : float, optional
            controls weighting of dice and cross-entropy loss, by default 0.5.
        beta : float, optional
            beta > 0.5 penalises false negatives more than false positives, \
            by default 0.5.
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = _EPSILON,
        ignore_index: int = None,
        class_weights=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.dice = DiceCoefficient(ignore_index=self.ignore_index)
        self.class_weights = class_weights

    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_weights=None) -> torch.Tensor:
        if type(class_weights) == type(None):
            class_weights = self.class_weights
    
        dice = torch.mean(_dice_similarity_c(y_pred, y_true, smooth=self.smooth))#,class_weights=class_weights))
        if y_pred.shape[1] > 1:
            y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
        else:
            y_pred = torch.nn.functional.log_sigmoid(y_pred, dim=1)

        cross_entropy = -y_true * y_pred#torch.log(y_pred + _EPSILON)

        if self.beta is not None:
            cross_entropy = self.beta * cross_entropy + (1 - self.beta) * cross_entropy

        if type(class_weights) != type(None):
            #class_weights = class_weights / (torch.sum(class_weights) * class_weights.shape[0])
            cross_entropy = torch.einsum('j,ij->ij', class_weights, cross_entropy)

        # sum over classes
        cross_entropy = torch.mean(torch.sum(cross_entropy, dim=1))
        if self.alpha is not None:
            return (self.alpha * cross_entropy) - ((1 - self.alpha) * dice)
        else:
            return cross_entropy - dice


#################################
# Symmetric Focal Tversky loss  #
#################################
class SymmetricFocalTverskyLoss(_Loss):

    """This is the implementation for binary segmentation.

    Args:
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7.
        gamma : float, optional
            focal parameter controls degree of down-weighting of easy examples,
            by default 0.75.
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001.
        common_class_index : int, optional
            index of the common class, by default 0.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = _EPSILON,
        common_class_index: int = 0,
        ignore_index: int = None,
        #class_weights=None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.common_class_index = common_class_index
        self.ignore_index = ignore_index
        #self.class_weights=class_weights

    #def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_weights=None) -> torch.Tensor:
    #    if type(class_weights) == type(None):
    #        class_weights = self.class_weights
    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Calculate Dice score
        tversky_class = _tversky_index_c(
            y_pred,
            y_true,
            alpha=self.delta,
            beta=1 - self.delta,
            smooth=self.smooth,
            reduction="none",
            #class_weights = class_weights,
        )

        n, c = y_pred.shape
        mask = torch.zeros_like(y_true, dtype=torch.bool)
        mask[:, self.common_class_index] = True

        back_tversky = tversky_class[mask].reshape(n, 1)
        back_tversky = (1 - back_tversky) * torch.pow(
            (1 - back_tversky + _EPSILON), -self.gamma
        )
        fore_tversky = tversky_class[~mask].reshape(n, c - 1)
        fore_tversky = (1 - fore_tversky) * torch.pow(
            (1 - fore_tversky + _EPSILON), -self.gamma
        )

        # Average class scores
        return torch.mean(torch.concat([back_tversky, fore_tversky], dim=1))


#################################
# Asymmetric Focal Tversky loss #
#################################
class AsymmetricFocalTverskyLoss(_Loss):
    """This is the implementation for binary segmentation.

    Args:
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            focal parameter controls degree of down-weighting of easy examples,
            by default 0.75.
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001.
        common_class_index : int, optional
            index of the common class, by default 0.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = _EPSILON,
        common_class_index: int = 0,
        ignore_index: int = None,
        #class_weights = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.common_class_index = common_class_index
        self.ignore_index = ignore_index
        #self.class_weights = class_weights

    #def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_weights=None) -> torch.Tensor:
    #    if type(class_weights) == type(None):
    #        class_weights = self.class_weights
    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        tversky_class = _tversky_index_c(
            y_pred,
            y_true,
            alpha=self.delta,
            beta=self.delta,
            smooth=self.smooth,
            reduction="none",
            #class_weights = class_weights,
        )

        n, c = y_pred.shape
        mask = torch.zeros_like(y_true, dtype=torch.bool)
        mask[:, self.common_class_index] = True

        back_tversky = 1 - tversky_class[mask].reshape(n, 1)
        fore_tversky = tversky_class[~mask].reshape(n, c - 1)
        fore_tversky = (1 - fore_tversky) * torch.pow(
            (1 - fore_tversky + _EPSILON), -self.gamma
        )

        # Average class scores
        return torch.mean(torch.concat([back_tversky, fore_tversky], dim=1))


################################
#    Symmetric Focal loss      #
################################
class SymmetricFocalLoss(_Loss):
    """
    Args:
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7.
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of
            easy examples, by default 0.75.
        smooth : float, optional
            smoothing constant to prevent division by zero errors, by default 0.000001.
        common_class_index : int, optional
            index of the common class, by default 0.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = _EPSILON,
        common_class_index: int = 0,
        ignore_index: int = None,
        class_weights = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.common_class_index = common_class_index
        self.ignore_index = ignore_index
        self.class_weights=class_weights

    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_weights=None) -> torch.Tensor:
        if type(class_weights) == type(None):
            class_weights = self.class_weights
    
        if y_pred.shape[1] > 1:
            y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
        else:
            y_pred = torch.nn.functional.log_sigmoid(y_pred, dim=1)
    
        cross_entropy = -y_true * y_pred

        n, c = y_pred.shape
        mask = torch.zeros_like(y_true, dtype=torch.bool)
        mask[:, self.common_class_index] = True

        back_pred = y_pred.exp()[mask].reshape(n, 1)
        back_ce = cross_entropy[mask].reshape(n, 1)
        back_ce = (
            (1 - self.delta) * torch.pow(1 - back_pred + _EPSILON, self.gamma) * back_ce
        )

        fore_pred = y_pred.exp()[~mask].reshape(n, c - 1)
        fore_ce = cross_entropy[~mask].reshape(n, c - 1)
        fore_ce = self.delta * torch.pow(1 - fore_pred, self.gamma) * fore_ce

        focal_loss = torch.concat([back_ce, fore_ce], dim=1)

        if type(class_weights) != type(None):
            #class_weights = class_weights / (torch.sum(class_weights) * class_weights.shape[0])
            focal_loss = torch.einsum('j,ij->ij', class_weights, focal_loss)


        return torch.mean(torch.sum(focal_loss, dim=1))


################################
#     Asymmetric Focal loss    #
################################
class AsymmetricFocalLoss(_Loss):
    """For Imbalanced datasets

    Args:
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7.
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of
            easy examples, by default 0.75.
        common_class_index : int, optional
            index of the common class, by default 0.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        common_class_index: int = 0,
        ignore_index: int = None,
        class_weights = None,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.common_class_index = common_class_index
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def _calc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, class_weights=None) -> torch.Tensor:
        if type(class_weights) == type(None):
            class_weights = self.class_weights
    
        if y_pred.shape[1] > 1:
            y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
        else:
            y_pred = torch.nn.functional.log_sigmoid(y_pred, dim=1)
    
        cross_entropy = -y_true * y_pred

        n, c = y_pred.shape
        mask = torch.zeros_like(y_true, dtype=torch.bool)
        mask[:, self.common_class_index] = True

        back_pred = y_pred.exp()[mask].reshape(n, 1)
        back_ce = cross_entropy[mask].reshape(n, 1)
        back_ce = (
            (1 - self.delta) * torch.pow(1 - back_pred + _EPSILON, self.gamma) * back_ce
        )

        fore_ce = self.delta * cross_entropy[~mask].reshape(n, c - 1)

        focal_loss = torch.concat([back_ce, fore_ce], dim=1)

        if type(class_weights) != type(None):
            #class_weights = class_weights / (torch.sum(class_weights) * class_weights.shape[0])
            focal_loss = torch.einsum('j,ij->ij', class_weights, focal_loss)

        return torch.mean(torch.sum(focal_loss, dim=1))


###########################################
#      Symmetric Unified Focal loss       #
###########################################
class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based
        and cross entropy-based loss functions into a single framework.

    Args:
        mu : float, optional
            represents lambda parameter and controls weight given to symmetric Focal
            Tversky loss and symmetric Focal loss, by default 0.5.
        delta : float, optional
            controls weight given to each class, by default 0.7.
        gamma : float, optional
            focal parameter controls the degree of background suppression and foreground
            enhancement, by default 0.75.
        common_class_index : int, optional
            index of the common class, by default 0.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self,
        mu: float = 0.5,
        delta: float = 0.7,
        gamma: float = 0.75,
        common_class_index: int = 0,
        ignore_index: int = None,
        class_weights=None,
    ):
        super().__init__()
        self.mu = mu
        self.class_weights = class_weights

        self.symmetric_ftl = SymmetricFocalTverskyLoss(
            delta=delta,
            gamma=gamma,
            common_class_index=common_class_index,
            ignore_index=ignore_index,
            #class_weights = class_weights
        )
        self.symmetric_fl = SymmetricFocalLoss(
            delta=delta,
            gamma=gamma,
            common_class_index=common_class_index,
            ignore_index=ignore_index,
            class_weights = class_weights
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,class_weights=None) -> torch.Tensor:
        """Calculate loss.

        Args:
            y_pred : Tensor of shape (batch_size, num_classes, ...).
                Predicted probabilities for each output class.
            y_true : Ground truth labels Tensor of shape (batch_size, ...).

        Returns:
            loss : Loss value.
        """
        if type(class_weights) == type(None):
            class_weights = self.class_weights

        symmetric_ftl = self.symmetric_ftl(y_pred, y_true)#, class_weights=class_weights)
        symmetric_fl = self.symmetric_fl(y_pred, y_true, class_weights=class_weights)
        if self.mu is not None:
            return (self.mu * symmetric_ftl) + ((1 - self.mu) * symmetric_fl)
        else:
            return symmetric_ftl + symmetric_fl


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
class AsymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based
        and cross entropy-based loss functions into a single framework.

    Args:
        mu : float, optional
            represents lambda parameter and controls weight given to asymmetric Focal
            Tversky loss and asymmetric Focal loss, by default 0.5.
        delta : float, optional
            controls weight given to each class, by default 0.7
        gamma : float, optional
            focal parameter controls the degree of background suppression and foreground
            enhancement, by default 0.75.
        common_class_index : int, optional
            index of the common class, by default 0.
        ignore_index : int, optional
            index of the ignore class, by default None.
    """

    def __init__(
        self,
        mu: float = 0.5,
        delta: float = 0.7,
        gamma: float = 0.75,
        common_class_index: int = 0,
        ignore_index: int = None,
        class_weights=None,
    ):
        super().__init__()
        self.mu = mu
        self.class_weights = class_weights

        self.asymmetric_ftl = AsymmetricFocalTverskyLoss(
            delta=delta,
            gamma=gamma,
            common_class_index=common_class_index,
            ignore_index=ignore_index,
            #class_weights=class_weights,
        )
        self.asymmetric_fl = AsymmetricFocalLoss(
            delta=delta,
            gamma=gamma,
            common_class_index=common_class_index,
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,class_weights=None) -> torch.Tensor:
        """Calculate loss.

        Args:
            y_pred : Tensor of shape (batch_size, num_classes, ...).
                Predicted probabilities for each output class.
            y_true : Ground truth labels Tensor of shape (batch_size, ...).

        Returns:
            loss : Loss value.
        """
        if type(class_weights) == type(None):
            class_weights = self.class_weights

        asymmetric_ftl = self.asymmetric_ftl(y_pred, y_true)#, class_weights=class_weights)
        asymmetric_fl = self.asymmetric_fl(y_pred, y_true, class_weights=class_weights)
        if self.mu is not None:
            return (self.mu * asymmetric_ftl) + ((1 - self.mu) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl
        


def lovasz_softmax_loss(pred: Tensor, target: Tensor, class_weights: Optional[Tensor] = None) -> Tensor:
    r"""Criterion that computes a surrogate multi-class intersection-over-union (IoU) loss.

    According to [1], we compute the IoU as follows:

    .. math::

        \text{IoU}(x, class) = \frac{|X \cap Y|}{|X \cup Y|}

    [1] approximates this fomular with a surrogate, which is fully differentable.

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the long tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{IoU}(x, class)

    Reference:
        [1] https://arxiv.org/pdf/1705.08790.pdf

    .. note::
        This loss function only supports multi-class (C > 1) labels. For binary
        labels please use the Lovasz-Hinge loss.

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
        class_weights: weights for classes with shape :math:`(num\_of\_classes,)`.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = lovasz_softmax_loss(pred, target)
        >>> output.backward()
    """
    KORNIA_CHECK_SHAPE(pred, ["B", "N", "H", "W"])

    KORNIA_CHECK_SHAPE(target, ["B", "H", "W"])

    if not pred.shape[1] > 1:
        raise ValueError(f"Invalid pred shape, we expect BxNxHxW, with N > 1. Got: {pred.shape}")

    if not pred.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")

    if not pred.device == target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")

    num_of_classes = pred.shape[1]
    # compute the actual dice score
    if class_weights is not None:
        KORNIA_CHECK_IS_TENSOR(class_weights, "class_weights must be Tensor or None.")
        KORNIA_CHECK(
            (class_weights.shape[0] == num_of_classes and class_weights.numel() == num_of_classes),
            f"weight shape must be (num_of_classes,): ({num_of_classes},), got {class_weights.shape}",
        )
        KORNIA_CHECK(
            class_weights.device == pred.device,
            f"class_weights and pred must be in the same device. Got: {class_weights.device} and {pred.device}",
        )

    # flatten pred [B, C, -1] and target [B, -1] and to float
    pred_flatten: Tensor = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target_flatten: Tensor = target.reshape(target.shape[0], -1)

    # get shapes
    B, C, N = pred_flatten.shape

    # compute softmax over the classes axis
    pred_soft: Tensor = pred_flatten.softmax(1)

    # compute actual loss
    foreground: Tensor = (
        torch.nn.functional.one_hot(target_flatten.to(torch.int64), num_classes=C).permute(0, 2, 1).to(pred.dtype)
    )
    errors: Tensor = (pred_soft - foreground).abs()
    errors_sorted, permutations = torch.sort(errors, dim=2, descending=True)
    batch_index = torch.arange(B, device=pred.device).unsqueeze(1).unsqueeze(2).expand(B, C, N)
    target_sorted = target_flatten[batch_index, permutations]
    target_sorted_sum = target_sorted.sum(2, keepdim=True)
    intersection = target_sorted_sum - target_sorted.cumsum(2)
    union = target_sorted_sum + (1.0 - target_sorted).cumsum(2)
    gradient = 1.0 - intersection / union
    if N > 1:
        gradient[..., 1:] = gradient[..., 1:] - gradient[..., :-1]
    weighted_errors = errors_sorted * gradient
    loss_per_class = weighted_errors.sum(2).mean(0)
    if class_weights is not None:
        loss_per_class *= class_weights
    final_loss: Tensor = loss_per_class.mean()
    return final_loss


class LovaszSoftmaxLoss(nn.Module):
    r"""Criterion that computes a surrogate multi-class intersection-over-union (IoU) loss.

    According to [1], we compute the IoU as follows:

    .. math::

        \text{IoU}(x, class) = \frac{|X \cap Y|}{|X \cup Y|}

    [1] approximates this fomular with a surrogate, which is fully differentable.

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the binary tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{IoU}(x, class)

    Reference:
        [1] https://arxiv.org/pdf/1705.08790.pdf

    .. note::
        This loss function only supports multi-class (C > 1) labels. For binary
        labels please use the Lovasz-Hinge loss.

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes > 1.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
        class_weights: weights for classes with shape :math:`(num\_of\_classes,)`.

    Return:
        a scalar with the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = LovaszSoftmaxLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(self, class_weights: Optional[Tensor] = None) -> None:
        super().__init__()
        self.class_weights = class_weights

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return lovasz_softmax_loss(pred=pred, target=target, class_weights=self.class_weights)