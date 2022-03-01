import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from collections import Counter


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


# class FocalLoss2d(nn.Module):
#     def __init__(self, gamma=2, alpha=0.25):
#         super(FocalLoss2d, self).__init__()
#         self.loss_fn = nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = self.loss_fn.reduction  # mean, sum, etc..

#     def forward(self, pred, true):
#         bceloss = self.loss_fn(pred, true)

#         pred_prob = torch.sigmoid(
#             pred
#         )  # p  pt는 p가 true 이면 pt = p / false 이면 pt = 1 - p
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # add balance
#         modulating_factor = torch.abs(true - pred_prob) ** self.gamma  # focal term
#         loss = alpha_factor * modulating_factor * bceloss  # bceloss에 이미 음수가 들어가 있음

#         if self.reduction == "mean":
#             return loss.mean()

#         elif self.reduction == "sum":
#             return loss.sum()

#         else:  # 'none'
#             return loss


# class FocalLoss2d(nn.Module):
#     def __init__(self, gamma=2, alpha=0.25):
#         super(FocalLoss2d, self).__init__()
#         self._gamma = gamma
#         self._alpha = alpha

#     def forward(self, y_true, y_pred):
#         cross_entropy_loss = torch.nn.BCELoss(y_true, y_pred)
#         p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
#         modulating_factor = 1.0
#         if self._gamma:
#             modulating_factor = torch.pow(1.0 - p_t, self._gamma)
#         alpha_weight_factor = 1.0
#         if self._alpha is not None:
#             alpha_weight_factor = y_true * self._alpha + (1 - y_true) * (
#                 1 - self._alpha
#             )
#         focal_cross_entropy_loss = (
#             modulating_factor * alpha_weight_factor * cross_entropy_loss
#         )
#         return focal_cross_entropy_loss.mean()
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2):
        # super(FocalLoss2d, self).__init__(weight, reduction)
        nn.Module.__init__(self)
        self.gamma = gamma

    def forward(self, input, target):
        # print(input.shape)
        # print(target.shape)

        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        logit = input.reshape(-1)
        target = target.reshape(-1)
        prob = torch.sigmoid(logit)
        prob = torch.where(target >= 0.5, prob, 1 - prob)
        logp = -torch.log(torch.clamp(prob, 1e-4, 1 - 1e-4))
        loss = logp * ((1 - prob) ** self.gamma)
        loss = 8 * loss.mean()
        return loss


# def focal_binary_cross_entropy(logits, targets, gamma=2):
#     l = logits.reshape(-1)
#     t = targets.reshape(-1)
#     p = torch.sigmoid(l)
#     p = torch.where(t >= 0.5, p, 1 - p)
#     logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
#     loss = logp * ((1 - p) ** gamma)
#     loss = num_label * loss.mean()
#     return loss


# class FocalLoss2d(nn.modules.loss._WeightedLoss):
#     def __init__(
#         self, gamma=2, weight=None, reduction="mean", balance_param=1,
#     ):
#         # super(FocalLoss2d, self).__init__(weight, reduction)
#         nn.Module.__init__(self)
#         self.gamma = gamma
#         # self.weight = weight
#         self.balance_param = balance_param
#         self.reduction = reduction

#     def forward(self, input, target):
#         # print(input.shape)
#         # print(target.shape)

#         # inputs and targets are assumed to be BatchxClasses
#         assert len(input.shape) == len(target.shape)
#         assert input.size(0) == target.size(0)
#         assert input.size(1) == target.size(1)

#         # weight = Variable(self.weight)

#         # compute the negative likelyhood
#         logpt = -F.binary_cross_entropy_with_logits(
#             input, target, reduction=self.reduction
#         )
#         pt = torch.exp(-logpt)

#         # compute the loss
#         focal_loss = ((1 - pt) ** self.gamma) * logpt
#         balanced_focal_loss = self.balance_param * focal_loss
#         return balanced_focal_loss


# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.logsigmoid()  # (dim=self.dim) # log_softmax
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # print(y_pred.ndim)

        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(
            torch.float32
        )  # 여기서one hot으로 만들어서 하는구나.. 그럼 괜찮지..
        y_pred = F.logsigmoid(  # softmax
            y_pred  # , dim=1
        )  # softmax를 loss전에 하도록 구성하기도 하는구나 그렇지 사실 model에 꼭 필요한건아니지 어디 넣어도 상관없을 듯

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)

        soft_f1 = 2 * tp / (2 * tp + fn + fp + self.epsilon)
        cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = cost.mean()
        return macro_cost

        # return 1 - f1.mean()


def calculate_prior(
    num_classes,
    img_max=None,
    prior=None,
    prior_txt=None,
    reverse=False,
    return_num=False,
):
    if prior_txt:
        labels = []
        with open(prior_txt) as f:
            for line in f:
                labels.append(int(line.split()[1]))
        occur_dict = dict(Counter(labels))
        img_num_per_cls = [occur_dict[i] for i in range(num_classes)]
    else:
        img_num_per_cls = []
        for cls_idx in range(num_classes):
            if reverse:
                num = img_max * (
                    prior ** ((num_classes - 1 - cls_idx) / (num_classes - 1.0))
                )
            else:
                num = img_max * (prior ** (cls_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    img_num_per_cls = torch.Tensor(img_num_per_cls)

    if return_num:
        return img_num_per_cls
    else:
        return img_num_per_cls / img_num_per_cls.sum()


# https://github.com/Joonsun-Hwang/imbalance-loss-test/blob/main/Loss%20Test.ipynb
class LADELoss(nn.Module):
    def __init__(
        self, num_classes=18, img_max=512, prior=0.1, prior_txt=None, remine_lambda=0.1,
    ):
        super().__init__()
        if img_max is not None or prior_txt is not None:
            self.img_num_per_cls = (
                calculate_prior(num_classes, img_max, prior, prior_txt, return_num=True)
                .float()
                .cuda()
            )
            self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        else:
            self.prior = None

        self.balanced_prior = torch.tensor(1.0 / num_classes).float().cuda()
        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        self.cls_weight = (
            self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float())
        ).cuda()

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(
            x_p, x_q, num_samples_per_cls
        )
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        per_cls_pred_spread = y_pred.T * (
            target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target)
        )  # C x N
        pred_spread = (
            y_pred
            - torch.log(self.prior + 1e-9)
            + torch.log(self.balanced_prior + 1e-9)
        ).T  # C x N

        num_samples_per_cls = torch.sum(
            target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1
        ).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(
            per_cls_pred_spread, pred_spread, num_samples_per_cls
        )

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss


_criterion_entrypoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "f1": F1Loss,
    "LADE": LADELoss,
    "focal_2": FocalLoss2d,
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion
