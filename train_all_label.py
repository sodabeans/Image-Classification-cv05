import argparse
from cProfile import label
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchsampler import ImbalancedDatasetSampler
from dataset import BaseTransform, CustomAugmentation, CutmixFace


from torch.utils.tensorboard import SummaryWriter
import wandb

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import MaskBaseDataset
from loss import create_criterion
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from adamp import AdamP


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ]  # 여기 잘 모르겠음. match하는 것을 찾는 것 같음
        i = [int(m.groups()[0]) for m in matches if m]  # match한게 있다면 거기서 하나 뽑아서 저렇게 하면
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def decode_multi_label(multi_class_label):
    # print(multi_class_label)
    multi_class_label = multi_class_label.to(torch.int16)
    mask_label = multi_class_label[:3]
    # print(mask_label)
    gender_label = multi_class_label[3:5]
    age_label = multi_class_label[5:]
    return (
        torch.argmax(mask_label) * 6
        + torch.argmax(gender_label) * 3
        + torch.argmax(age_label) * 1
    )


# 결국은 model을 저장해주는 path를 만들기 위한 코드임


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_module = getattr(
        import_module("dataset"),
        args.dataset,  # iomport module로 현재 디렉토리의 dataset모듈을 불러오는거 ㅇㅇ 거기의 BaseAugmenation을 가져오는거
    )  # default: MaskBaseDataset
    dataset = dataset_module(data_dir=data_dir, target_type=args.target_type,)
    num_classes = dataset.num_labels  # 18

    # transform
    transform = BaseTransform(
        resize=args.resize, mean=dataset.mean, std=dataset.std, cutmix=args.cutmix
    )
    dataset.set_transform(transform)

    # -- augmentation
    if args.augmentation == "CustomAugmentation":
        augmentation_module = getattr(
            import_module("dataset"), args.augmentation
        )  # default: BaseAugmentation
        augmentation = augmentation_module()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(num_classes=num_classes, dropout=0.7).to(device)
    model = torch.nn.DataParallel(model)
    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = AdamP(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
        nesterov=True,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=args.lr_decay_step,
        threshold=0.001,
        verbose=True,
        min_lr=1e-5,
        threshold_mode="abs",
    )
    for epoch in range(args.epochs):
        # train loop
        epoch += 1
        model.train()
        loss_value = 0
        f1 = 0
        acc = 0
        gc.collect()
        torch.cuda.empty_cache()
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # https://discuss.pytorch.org/t/how-to-augment-the-minority-class-only-in-an-unbalanced-dataset/13797/2
            if np.random.random() > 0.5 and args.cutmix == "yes":
                inputs, labels = CutmixFace(0.8)(inputs, labels)

            not_list = [
                [1, 0, 0, 1, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 1, 0, 0],
                [1, 0, 0, 0, 1, 0, 1, 0],
            ]
            if args.augmentation == "CustomAugmentation":
                aug_inputs = []
                for input, label in zip(inputs, labels):
                    if (
                        list(label.cpu().detach().numpy()) not in not_list
                    ):  # https://velog.io/@xdfc1745/Bool-value-of-Tensor-with-more-than-one-value-is-ambiguous-in-Pytorch
                        aug_input = augmentation(input)
                        aug_inputs.append(aug_input)
                    else:
                        aug_inputs.append(input)
                inputs = torch.stack(aug_inputs)
            optimizer.zero_grad()
            outs = model(inputs)
            preds = torch.sigmoid(outs).data > 0.5
            loss = criterion(outs, labels)
            loss.backward()
            # print(preds)
            # # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            loss_value += loss.item()
            # matches += (preds == labels).sum().item()
            pred_metric = torch.clone(preds).detach().cpu().numpy()
            label_metric = torch.clone(labels).detach().cpu().numpy()
            # print(f1_score(label_f1, pred_f1, average="macro"))
            acc += accuracy_score(label_metric, pred_metric)
            f1 += f1_score(label_metric, pred_metric, average="macro")
            if (
                idx + 1
            ) % args.log_interval == 0:  # 적당 수준 돌았을 때 배치가 어느정도 우리가 설정한거에 맞게 돌면 찍는거 이 기준으로 평균도
                train_loss = (
                    loss_value / args.log_interval
                )  # 일종의 평균 로스 ( 로스를 다 찍는게 아닌 배치 평균찍는거-> 좀 더 생각)
                train_acc = acc / args.log_interval
                current_lr = get_lr(optimizer)  # lr scheduler를 쓰기 때문에 사용.
                train_f1 = f1 / args.log_interval
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "  # 총 몇 에폭중 몇 에폭째인지 총 배치 중에 몇번쨰 배치인지
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || train f1-score {train_f1:4.4}"
                )
                loss_value = 0
                acc = 0
                f1 = 0
    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    import warnings

    warnings.filterwarnings(action="ignore")

    from dotenv import load_dotenv
    import gc
    import os

    gc.collect()
    torch.cuda.empty_cache()
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskSplitByProfileDataset",
        help="dataset augmentation type (default: MaskSplitByProfileDataset)",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="multiclass",
        help="dataset target type (default: multiclass)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="CustomAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--cutmix", type=str, default="no", help="data cutmix yes or no (default: no)",
    )
    parser.add_argument(
        "--resize",
        nargs="+",
        type=int,
        default=[512, 384],
        help="resize size for image when training (default : 512,384)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=64,
        help="input batch size for validing (default: 64)",  # https://study-grow.tistory.com/entry/RuntimeError-cuDNN-error-CUDNNSTATUSNOTSUPPORTED-This-error-may-appear-if-you-passed-in-a-non-contiguous-input-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0%EB%B2%95
    )  # 이거 줄이니까 문제 해결
    parser.add_argument(
        "--model",
        type=str,
        default="Res18Model",
        help="model type (default: Res18Model)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer type (default: Adam)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion", type=str, default="f1", help="criterion type (default: f1)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=3,
        help="learning rate scheduler deacy step (default: 3)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="all", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # sampler
    parser.add_argument(
        "--sampler", default="Weight", help="how to sampling ( default: weight) "
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"),
    )  # 환경변수 설정 & 없다면 기본값 설정.
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    args = parser.parse_args()

    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
