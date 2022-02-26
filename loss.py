import torch
import torch.nn as nn
import torch.nn.functional as F

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
        log_prob = F.logsigmoid(input_tensor, dim=-1)  # log_softmax
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.logsigmoid(dim=self.dim) # log_softmax
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
        y_pred = F.logsigmoid(   # softmax
            y_pred, dim=1
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
