#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable, Function
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np
from soft_skeleton import SoftSkeletonize

#ERnet
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """

    input = input.unsqueeze(0)
    shape = np.array(input.shape)  # (1,1,64,208,224)

    shape[1] = num_classes  ### shape[1]==2

    shape = tuple(shape)
    result = torch.zeros(shape)

    result = result.scatter_(1, input.cpu() / 255, 1).cuda()  #
    return result


class DiceLoss_v2(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss_v2, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        target = target.type(torch.LongTensor).cuda()
        # target = make_one_hot(target, num_classes=predict.shape[1]) ###
        target = target.cuda()
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


# def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
#     # assumes that input is a normalized probability
#     assert input.size == target.size, "'input' and 'target' must have hte same size"
#     if ignore_index is not None:
#         mask = target.clone().ne_(ignore_index)
#         mask.requires_grad = False
#         input = input * mask
#         target = target * mask
#
#     input = flatten(input)
#     target = flatten(target)
#
#     target = target.float()
#     # compute per channel dice
#     intersect = (input * target).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect
#
#     denominator = (input + target).sum(-1)
#     return 2.0 * intersect / denominator.clamp(min=epsilon)
#
#
# class DiceLoss(nn.Module):
#     def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=False,
#                  skip_last_target=False):
#         super(DiceLoss, self).__init__()
#         self.epsilon = epsilon
#         self.register_buffer('weight', weight)
#         self.ignore_index = ignore_index
#         if sigmoid_normalization:
#             self.normalization = nn.Sigmoid()
#         else:
#             self.normalization = nn.Softmax(dim=1)
#
#         self.skip_last_target = skip_last_target
#
#     def forward(self, input, target):
#         input = self.normalization
#         if self.weight is not None:
#             weight = Variable(self.weight, requires_grad=False)
#         else:
#             weight = None
#
#         if self.skip_last_target:
#             target = target[:, :-1, ...]
#
#         per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
#                                                     weight=weight)
#         return torch.mean(1.0 - per_channel_dice)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1).float()

        num = torch.sum(torch.mul(predict, target)) * 2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # target = torch.LongTensor(target)
        # target = make_one_hot(target, num_classes=predict.shape[1])   ###   num_classes=predict.shape[1]
        target = target.cuda()
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


# ---------------------------------------------------------------------------------------------------------
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        # target = target.type(torch.LongTensor).cuda()
        # target = make_one_hot(target, 2)
        self.save_for_backward(input, target)
        eps = 1
        # eps = 1

        # dot是返回两个矩阵的点集
        # inter,uniun:两个值的大小分别是10506.6,164867.2
        # print('input,target:', input, np.max()target)
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        # print('self.inter, self.union:', self.inter, self.union)

        # print("inter,uniun:",self.inter,self.union)

        t = (2 * self.inter.float()) / self.union.float()
        # print('DiceCoeff:', t)
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        # 这里没有打印出来，难道没有执行到这里吗
        # print("grad_input, grad_target:",grad_input, grad_target)

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # print("size of input, target:", input.shape, target.shape)

    for i, c in enumerate(zip(input, target)):
        # c[0],c[1]的大小都是原图大小torch.Size([1, 576, 544])
        # print("size of c0 c1:", c[0].shape,c[1].shape)
        s = s + DiceCoeff().forward(c[0], c[1])

    # print(s, i + 1, s / (i + 1))

    return s / (i + 1)


def dice_coeff_loss(input, target):
    return 1 - dice_coeff(input, target)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        class_weights = self._class_weights(inputs)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(torch.cat((1. - inputs, inputs), 1), target, weight=class_weights,
                               ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(inputs):
        # normalize the input first
        # inputs = F.softmax(inputs)
        flattened = flatten(torch.cat((1. - inputs, inputs), 1))
        # print(flattened)
        # print(inputs.shape, inputs.dtype)
        # print(flattened.shape,flattened.dtype)
        # input("wait..")
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


# ---------------------------------------------------------------------------------------------

def log_sum_exp(x):
    # b is a shift factor to avoid overflow
    # x.size() = [N,C]
    b, _ = torch.max(x, 1)
    y = b + torch.log(torch.exp(x - b.expand_as(x)).sum(1))
    return y.squeeze(1)


def class_select(logits, target):
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(
            torch.arange(0, num_classes).long().repeat(batch_size, 1).cuda(device).eq(
                target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(
            torch.arange(0, num_classes).long().repeat(batch_size, 1).eq(target.data.repeat(num_classes, 1).t()))

    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = log_sum_exp(logits) - class_select(logits, target)
    if weights is not None:
        assert list(loss.size()) == list(weights.size())
        loss = loss * weights
    return loss


class WeightCELoss(nn.Module):
    def __init__(self, aggregate='mean'):
        super(WeightCELoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: clipped tensor
    """
    t = t.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max

    return result


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, size_average=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, gt):
        assert pred.size() == gt.size() and pred.size()[1] == 1
        pred_oh = torch.cat((pred, 1.0 - pred), dim=1)  # [b, 2, h, w]
        gt_oh = torch.cat((gt, 1.0 - gt), dim=1)  # [b, 2, h, w]
        pt = (gt_oh * pred_oh).sum(1)  # [b, h, w]
        focal_map = - self.alpha * torch.pow(1.0 - pt, self.gamma) * torch.log2(
            clip_by_tensor(pt, 1e-12, 1.0))  # [b, h, w]

        if self.size_average:
            loss = focal_map.mean()
        else:
            loss = focal_map.sum()
        return loss


# ---------------------------------------------------------------------------------------------------
def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /  (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def re_DiceLoss(inputs, targets):
    smooth = 1
    input_flat = inputs.view(-1)
    target_float = targets.view(-1)
    intersection = input_flat * target_float
    unionsection = input_flat.pow(2).sum() + target_float.pow(2).sum() + smooth
    loss = unionsection / (2 * intersection.sum() + smooth)
    loss = loss.sum()
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        preds = preds.cuda()
        labels = labels.type(torch.LongTensor).squeeze(0).cuda()
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class boundary_loss_func(nn.Module):
    def __init__(self):
        super(boundary_loss_func, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor([1.]))
        self.weight2 = nn.Parameter(torch.Tensor([1.]))
        # self.FocalLoss = FocalLoss()
        # self.focal_loss = focal_loss()
        # self.DiceLoss = dice_coeff_loss()

    def forward(self, boundary_logits, gtmasks):
        laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 26, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1],
            dtype=torch.float32).reshape(1, 1, 3, 3, 3)
        boundary_targets = F.conv3d(gtmasks, laplacian_kernel.cuda(), padding=1)
        # pred_boundary = F.conv3d(boundary_logits, laplacian_kernel.cuda(), padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0
        # pred_boundary = pred_boundary.clamp(min=0)
        # pred_boundary[pred_boundary > 0.1] = 1
        # pred_boundary[pred_boundary <= 0.1] = 0
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_targets = F.interpolate(
                boundary_targets, boundary_logits.shape[2:], mode='nearest')
        dice_loss = dice_coeff_loss(boundary_logits, boundary_targets)
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
        total_loss = self.weight1.pow(-2) * bce_loss + \
                     self.weight2.pow(-2) * dice_loss + \
                     (1 + self.weight1 * self.weight2).log()  # return (2 * bce_loss + 8 * dice_loss) / 10

        # FocalLoss = self.focal_loss(pred_boundary, boundary_targets)
        # total_loss = self.weight1.pow(-2) * FocalLoss + \
        #              self.weight2.pow(-2) * dice_loss + \
        #              (1 + self.weight1 * self.weight2).log()  # return (2 * bce_loss + 8 * dice_loss) / 10
        # print('total_loss:', total_loss)
        return total_loss




def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if len(true.shape) == 2:
        true = true.unsqueeze(0).unsqueeze(0)
    elif len(true.shape) == 3:
        true = true.unsqueeze(1)

    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = nn.functional.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)
##本项目使用的
# def class2one_hot(seg, K=2):
#        # Breaking change but otherwise can't deal with both 2d and 3d
#        # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
#        #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]
#
#        assert sset(seg, list(range(K))), (self.uniq(seg), K)
#
#        img_shape = tuple(seg.shape)  # type: Tuple[int, ...]
#
#        device = seg.device
#        seg = seg.to(torch.int64)
#        # 创建onehot编码，背景10 前景01
#        res = torch.zeros((K, *img_shape), dtype=torch.int32, device=device).scatter_(0, seg[ None, ...], 1)
#
#        assert res.shape == (K, *img_shape)
#        assert one_hot(res)
#
#        return res
# def one_hot(t, axis=0) :
#     """
#     通过检查单纯性和元素是否为01判断是否为onehot编码
#     :param t:
#     :param axis:
#     :return:
#     """
#     return simplex(t, axis) and sset(t, [0, 1])
# def simplex(t, axis=0):
#     """
#     是用于检查一个张量在指定轴上是否满足单纯形条件的函数。在这里，单纯形条件指的是张量在该轴上的元素之和为1。
#     即判断是不是one-hot编码
#     :param t:
#     :param axis:
#     :return:
#     """
#     _sum = cast(Tensor, t.sum(axis).type(torch.float32))
#     _ones = torch.ones_like(_sum, dtype=torch.float32)
#     flag = torch.allclose(_sum, _ones)
#     return flag
#
# def sset(a, sub):
#     """
#     判断a中的元素是不是sub的子集，及通过检查元素是不是只包含0 1判断是不是onehot编码
#     :param a:
#     :param sub:
#     :return:
#     """
#
#     return self.uniq(a).issubset(sub)
#
# def uniq(a) :
#     """
#     提取a中包含的元素值
#     :param a:
#     :return:
#     """
#     return set(torch.unique(a.cpu()).numpy())

class Dice_Loss(nn.Module):
    def __init__(self,smooth=0.001):
        super(Dice_Loss,self).__init__()
        self.smooth = smooth

    def forward(self,pred,target):
        iflat = pred.view(-1)
        tflat = target.view(-1)

        instersection = torch.sum((iflat*tflat))
        return 1.0 - ((2.0*instersection+self.smooth)/(torch.sum(iflat)+torch.sum(tflat)+self.smooth))
class Boundary_Loss(nn.Module):
    def __init__(self):
        super(Boundary_Loss,self).__init__()
        self.idc = [1]

    def forward(self,pred, target):

        pc = pred[:, 0:, ...]
        dc = target[:, 1:, ...]

        multipled = torch.einsum("bkxyz,bkxyz->bkxyz", pc, dc)

        loss = multipled.mean()

        return loss
class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_pred, y_true):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice
#SAUnet
class totalLoss(nn.Module):
    def __init__(self,  lmbda=1.0, epsilon=1.0, alpha=1.0):
        super(totalLoss, self).__init__()

        # self.CELoss = nn.CrossEntropyLoss()
        # self.CELoss = nn.BCEWithLogitsLoss()
        # self.bce3d = nn.BCELoss()
        # self.bce3d = nn.BCEWithLogitsLoss()


        self.diceloss = Dice_Loss()
        self.skeloss = soft_cldice()

        self.edgedice = Dice_Loss()#将edge的交叉熵损失改为dice损失
        # self.edgeboundary = Boundary_Loss()#edge使用边界损失

        self.epsilon = epsilon
        self.lmbda = lmbda
        self.alpha = alpha


    def forward(self, prelabels,siglabels,prelabelEdge,prelabelEdges,labels,labelEdge=None,labelBoundaryMap = None,labelEdgeBoundaryMap = None):
        """

        :param prelabels: 预测结果，无使用sigmoid
        :param siglabels: sigmoid后的结果
        :param prelabelEdge: 预测的边缘，无使用sigmoid,dice为使用sigmoid
        :param labels: 标签真值
        :param labelEdge: 边缘标签的真值
        :return:
        """
        # seg
        labels = labels.unsqueeze(1)
        # if labelEdge.numel() != 0:
        if labelEdge is not None:
            labelEdge = labelEdge.unsqueeze(1)

        #-------------整体的损失----------------------
        #加权二元交叉熵
        weights = calWeights(labels)
        ce = F.binary_cross_entropy_with_logits(prelabels, labels, weight=weights)

        # dice损失
        dice = self.diceloss(siglabels,labels)

        #骨架线约束
        ske = self.skeloss(siglabels,labels)

        # ---------------edge的损失----------------------
        ##使用二元交叉熵
        weights = calWeights(labelEdge,w0=0.1)
        edgece = F.binary_cross_entropy_with_logits(prelabelEdge, labelEdge, weight=weights)

        ##使用dice
        edgedice = self.edgedice(prelabelEdges,labelEdge)

        ##使用boundary loss
        # edgebod = self.edgeboundary(prelabelEdge,labelEdgeBoundaryMap)


        return self.epsilon*dice + self.lmbda*ce +self.alpha* edgece+ edgedice+ske,[ce,dice,ske,edgece,edgedice ] # + 0.1*att
        # return self.epsilon * dice + self.lmbda * ce + self.alpha * edgedice + ske, [ce, dice, ske,edgedice]  # + 0.1*att

        # return  ce

def calWeights(label, w1=1.0, w0=0.5):
    weights = torch.ones_like(label)
    # totalnum = torch.numel(label)
    # ele1 = torch.count_nonzero(label==1).item()
    # if ele1 == 0:
    #     ele1 = int(totalnum * 0.1)
    # ele0 = totalnum - ele1
    # weights[label == 1] =  totalnum /ele1
    # weights[label == 0] =  totalnum / ele0
    weights[label == 1] =  w1
    weights[label == 0] =  w0

    # weights /= torch.sum(weights)
    return  weights


# boundary loss  https://github.com/LIVIAETS/boundary-loss/blob/master/losses.py
#!/usr/bin/env python3.9

# MIT License

# Copyright (c) 2023 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from typing import List, cast
#
# import torch
# import numpy as np
# from torch import Tensor, einsum
#
# from utils import simplex, probs2one_hot, one_hot
# from utils import one_hot2hd_dist


# class CrossEntropy():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         assert simplex(probs) and simplex(target)
#
#         log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
#         mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
#
#         loss = - einsum("bkwh,bkwh->", mask, log_p)
#         loss /= mask.sum() + 1e-10
#
#         return loss
#
#
# class GeneralizedDice():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         assert simplex(probs) and simplex(target)
#
#         pc = probs[:, self.idc, ...].type(torch.float32)
#         tc = target[:, self.idc, ...].type(torch.float32)
#
#         w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
#         intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
#         union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))
#
#         divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)
#
#         loss = divided.mean()
#
#         return loss
#
#
# class DiceLoss():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         assert simplex(probs) and simplex(target)
#
#         pc = probs[:, self.idc, ...].type(torch.float32)
#         tc = target[:, self.idc, ...].type(torch.float32)
#
#         intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
#         union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))
#
#         divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)
#
#         loss = divided.mean()
#
#         return loss

#
# class SurfaceLoss():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
#         assert simplex(probs)
#         assert not one_hot(dist_maps)
#
#         pc = probs[:, self.idc, ...].type(torch.float32)
#         dc = dist_maps[:, self.idc, ...].type(torch.float32)
#
#         multipled = einsum("bkwh,bkwh->bkwh", pc, dc)
#
#         loss = multipled.mean()
#
#         return loss
#
#
# BoundaryLoss = SurfaceLoss


# class HausdorffLoss():
#     """
#     Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
#     """
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         assert simplex(probs)
#         assert simplex(target)
#         assert probs.shape == target.shape
#
#         B, K, *xyz = probs.shape  # type: ignore
#
#         pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
#         tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
#         assert pc.shape == tc.shape == (B, len(self.idc), *xyz)
#
#         target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
#                                               for b in range(B)], axis=0)
#         assert target_dm_npy.shape == tc.shape == pc.shape
#         tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)
#
#         pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
#         pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
#                                             for b in range(B)], axis=0)
#         assert pred_dm_npy.shape == tc.shape == pc.shape
#         pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)
#
#         delta = (pc - tc)**2
#         dtm = tdm**2 + pdm**2
#
#         multipled = einsum("bkwh,bkwh->bkwh", delta, dtm)
#
#         loss = multipled.mean()
#
#         return loss
#
#
# class FocalLoss():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         self.gamma: float = kwargs["gamma"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         assert simplex(probs) and simplex(target)
#
#         masked_probs: Tensor = probs[:, self.idc, ...]
#         log_p: Tensor = (masked_probs + 1e-10).log()
#         mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
#
#         w: Tensor = (1 - masked_probs)**self.gamma
#         loss = - einsum("bkwh,bkwh,bkwh->", w, mask, log_p)
#         loss /= mask.sum() + 1e-10
#
#         return loss
