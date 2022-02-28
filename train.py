import argparse
import glob
import json
from math import gamma
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from sched import scheduler

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset import MaskBaseDataset
from loss import create_criterion
from sklearn.metrics import f1_score

from adamp import AdamP
from CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts

from albumentations import *
from albumentations.pytorch import ToTensorV2


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


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(  # 서브 플롯 크기 또는 간격을 개선
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


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


# 결국은 model을 저장해주는 path를 만들기 위한 코드임


def get_transforms(
    need=("train", "val"),
    img_size=(512, 384),
    mean=(0.548, 0.504, 0.479),
    std=(0.237, 0.247, 0.246),
):
    """
        train 혹은 validation의 augmentation 함수를 정의합니다. train은 데이터에 많은 변형을 주어야하지만, validation에는 최소한의 전처리만 주어져야합니다.
    
        Args:
            need: 'train', 혹은 'val' 혹은 둘 다에 대한 augmentation 함수를 얻을 건지에 대한 옵션입니다.
            img_size: Augmentation 이후 얻을 이미지 사이즈입니다.
            mean: 이미지를 Normalize할 때 사용될 RGB 평균값입니다.
            std: 이미지를 Normalize할 때 사용될 RGB 표준편차입니다.

        Returns:
            transformations: Augmentation 함수들이 저장된 dictionary 입니다. transformations['train']은 train 데이터에 대한 augmentation 함수가 있습니다.
        """
    transformations = {}
    if "train" in need:
        transformations["train"] = Compose(
            [
                Resize(img_size[0], img_size[1], p=1.0),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(
                    hue_shift_limit=0.2,
                    sat_shift_limit=0.2,
                    val_shift_limit=0.2,
                    p=0.5,
                ),
                RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                GaussNoise(p=0.5),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    if "val" in need:
        transformations["val"] = Compose(
            [
                Resize(img_size[0], img_size[1]),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    return transformations


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset   # https://technote.kr/249  # 기존에 존재하지 않는 속성을 가져오려하면서 기본갑싱 있는 경우
    # 즉 data
    # import_module => 모듈을 임포트 합니다. name 인자는 절대나 상대적인 항으로 임포트 할 모듈을 지정합니다
    # 출처: https://iosroid.tistory.com/118 [조용한 담장]
    # 결국 지정된 패키지나 모듈을 반환하는 것이다.
    # 여기서는 dataset을 반환하려고 하는데 이 때 이 dataset모듈에 args.dataset을 가져오는코드
    # 즉 이런식으로 augmentation을 설정해줄 수 있다고 보면 될 듯.
    # 이런 augmnetation적용된 데이터를 dataset에 할당
    dataset_module = getattr(
        import_module("dataset"),
        args.dataset,  # import module로 dataset.py모듈을 import하고 거기의 BaseAugmenation class을 가져오는거
    )  # default: MaskBaseDataset
    dataset = dataset_module(data_dir=data_dir,)
    num_classes = dataset.num_classes  # 18

    # # -- augmentation
    # transform_module = getattr(
    #     import_module("dataset2"), args.augmentation
    # )  # default: get_transforms
    # transforms = transform_module(mean=dataset.mean, std=dataset.std,)
    # 엇 아니네 여기가 진짜 transform이 일어나는 과정인 것 같은데,, 위에 써진 augmenatation내용이 여기꺼임.
    # dataset.set_transform(transform)

    transform = get_transforms()

    # -- data_loader
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_set.dataset.set_transform(transform["train"])
    val_set.dataset.set_transform(transform["val"])

    class_sample_counts = [
        2745,
        2050,
        415,
        3660,
        4085,
        545,
        549,
        410,
        83,
        732,
        817,
        109,
        549,
        410,
        83,
        732,
        817,
        109,
    ]

    class_weights = 18900.0 / torch.tensor(class_sample_counts, dtype=torch.float)
    labels = [t[1] for t in train_set]
    weights = [class_weights[labels[i]] for i in range(len(train_set))]

    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,  # WeightedRandomSampler 사용시 shuffle은 false
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,  # default = False
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(
        model
    )  # https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
    # multi GPU사용을 위한 설정

    # wandb
    wandb.watch(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    # optimizer = opt_module(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     weight_decay=5e-4,
    # )
    optimizer = AdamP(
        model.parameters(), lr=3e-5, weight_decay=1e-5, nesterov=True
    )  # 실험

    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.5,
    #     patience=args.lr_decay_step,
    #     threshold=0.001,
    #     verbose=True,
    #     min_lr=1e-5,
    #     threshold_mode="abs",
    # )
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=10, T_mult=1, eta_max=3e-4, T_up=3, gamma=1,
    )
    # T_0 : 최초 주기값,
    # T_mult는 주기가 반복되면서 최초 주기값에 비해 얼만큼 주기를 늘려나갈 것인지 스케일 값
    # eta_max는 learning rate의 최댓값
    # T_up은 Warm up 시 필요한 epoch 수를 지정하며 일반적으로 짧은 epoch 수를 지정
    # gamma는 주기가 반복될수록 eta_max 곱해지는 스케일값

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            vars(args), f, ensure_ascii=False, indent=4
        )  # 특정 json파일에 이러한 config를 적어준다.

    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf

    # early stopping
    patience = 6  # 5
    counter = 0

    for epoch in range(args.epochs):
        # train loop
        # epoch += 1
        model.train()
        loss_value = 0
        matches = 0
        f1 = 0
        gc.collect()
        torch.cuda.empty_cache()
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            pred_f1 = torch.clone(preds).detach().cpu().numpy()
            label_f1 = torch.clone(labels).detach().cpu().numpy()
            f1 += f1_score(label_f1, pred_f1, average="macro")
            if (idx + 1) % args.log_interval == 0:
                # 적당 수준 돌았을 때 배치가 어느정도 우리가 설정한거에 맞게 돌면 찍는거 이 기준으로 평균도
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)

                train_f1 = f1 / args.log_interval

                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/F1-score", train_f1, epoch * len(train_loader) + idx
                )

                loss_value = 0
                matches = 0
                f1 = 0

        scheduler.step()  # CosineAnnealingLR 시

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []

            # wandb
            example_imgaes = []

            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_pred_f1 = torch.clone(preds).detach().cpu().numpy()
                val_label_f1 = torch.clone(labels).detach().cpu().numpy()
                f1_item = f1_score(val_label_f1, val_pred_f1, average="macro")
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_f1_items.append(f1_item)

                # wandb
                example_imgaes.append(
                    wandb.Image(
                        inputs[0],
                        caption="Pred: {} Truth: {}".format(preds[0].item(), labels[0]),
                    )
                )

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    figure = grid_image(
                        inputs_np,
                        labels,
                        preds,
                        n=16,
                        shuffle=args.dataset != "MaskSplitByProfileDataset",
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 = np.sum(val_f1_items) / len(val_loader)

            # scheduler.step(val_f1)  # ReduceLROnPlateau 시

            best_val_loss = min(best_val_loss, val_loss)
            # if val_acc > best_val_acc:
            #     print(
            #         f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
            #     )
            #     torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
            #     best_val_acc = val_acc
            if val_f1 > best_val_f1:
                print(
                    f"New best model for val f1-score : {val_f1:4.4}! saving the best model.."
                )
                torch.save(
                    model.module.state_dict(), f"{save_dir}/best.pth"
                )  # best 모델이 계속 업데이트될 듯
                best_val_f1 = val_f1
                counter = 0
            else:
                counter += 1

            # if counter > patience:
            #     print("Early Stopping...")
            #     break

            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            # print(
            #     f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
            #     f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            # )
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} , F1-score : {val_f1:4.4}|| "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, Best F1-score : {best_val_f1:4.4}"
            )

            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)

            # wandb
            wandb.log(
                {
                    "Examples": example_imgaes,
                    "Valid Acc": val_acc,
                    "Valid Loss": val_loss,
                    "Valid F1-score": val_f1,
                }
            )
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import gc
    import os

    gc.collect()
    torch.cuda.empty_cache()
    load_dotenv(verbose=True)

    # wandb
    # wandb.init(project="Image_Classification", entity="jaeook", reinit=True)
    wandb.init(project="test_jaeook", entity="shine_light", reinit=True)
    wandb.run.name = "Efficientnet_b4_test9"

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
        default="MaskBaseDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="get_transforms",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs="+",
        type=list,
        default=[512, 384],
        help="resize size for image when training",
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
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer type (default: SGD)"
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
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=10,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/Data/train/images"),
    )  # "/opt/ml/input/data/train/images"
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    args = parser.parse_args()

    # wandb
    # wandb.run.name = 'args.name'
    wandb.config.update(args)

    wandb.run.save()

    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
