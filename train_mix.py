import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR,ExponentialLR, CosineAnnealingLR,ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss

from dataset import MaskBaseDataset
from loss import create_criterion
from sklearn.metrics import f1_score
import wandb
# from .radam import RAdam

def rand_bbox(size, lam):  # size : [Batch_size, Channel, Width, Height]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
    # cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # 패치의 중앙 좌표 값 cx, cy
    # cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 모서리 좌표 값
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(batch, alpha, vertical, vertical_half):
    data, targets = batch

    if vertical_half:
        lam = 0.5
    else:
        lam = np.random.beta(alpha, alpha)

    if not vertical:
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        image_h, image_w = data.shape[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
        targets = (targets, shuffled_targets, lam)

        return data, targets
    else:
        rand_index = torch.randperm(data.size(0))
        target_a = targets  # 원본 이미지 label
        target_b = targets[rand_index]  # 패치 이미지 label
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        targets = (target_a, target_b, lam)
        return data, targets


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
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n**0.5)
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
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    # wandb.init(project="hyunjin", entity="shine_light", config = {
    #     "learning_rate": 0.001,
    #     "epochs": 30,
    #     "batch_size": 64  
    # })
    # wandb.run.name = 'Vgg19-ExponentialLR-1'
    # # wandb.watch(model) 옵션
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    print("Does cuda run well?")
    if use_cuda:
        print("Yeeeaaahhhhhh")
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    # StepLR
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # CyclicLR
    # scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=4, step_size_down=6, mode='triangular',cycle_momentum=False)
    # scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=6, step_size_down=None, mode='triangular2')

    # ExponentialLR
    # scheduler = ExponentialLR(optimizer, gamma=0.95)

    # CosineAnnealingWarmRestarts
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=1, eta_min=0, last_epoch=-1)

    # Custom CosineAnnealingWarmRestarts
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.005,  T_up=6, gamma=0.5)

    # CosineAnnealingLR
    # scheduler = CosineAnnealingLR(optimizer, eta_min=0.001,T_max=3)

    # plateau
    # scheduler = ReduceLROnPlateau(optimizer,'min')


    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf
    for epoch in tqdm(range(args.epochs)):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        f1=0
        target_list = [[],[]]
        pred_target_list = []
        for idx, train_batch in enumerate(train_loader):
            train_batch = cutmix(train_batch, 1, vertical=False, vertical_half = np.False_)

            inputs, labels_cut = train_batch
            inputs = inputs.to(device)
            labels,target,lam = (labels_cut[0].to(device), labels_cut[1].to(device), labels_cut[2])
            target_list[0] += labels.tolist()
            target_list[1] += target.tolist()

            optimizer.zero_grad()

            outs = model(inputs)
            # preds = torch.argmax(outs, dim=-1)     
            preds = torch.softmax(outs, dim=-1)    
            loss_func =  CrossEntropyLoss(reduction='mean')
            loss = lam * loss_func(preds, labels) + (1 - lam) * loss_func(preds, target)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()
            # cutmix acc
            arg_preds = torch.argmax(preds, dim=1)
            correct1 = (arg_preds == labels).sum().item()
            correct2 = (arg_preds == target).sum().item()
            accuracy = (lam * correct1 + (1 - lam) * correct2)
            pred_target_list += arg_preds.tolist()
            # cutmix acc end
            loss_value += loss.item()
            matches += accuracy

            # add f1
            # print('preds',preds)
            # print('arg_preds',arg_preds)
            pred_f1 = torch.clone(arg_preds).detach().cpu().numpy()
            label_f1 = torch.clone(labels).detach().cpu().numpy()
            # print(f1_score(label_f1, pred_f1, average="macro"))

            # f1 += f1_score(label_f1, pred_f1, average="macro")
            # print('pred_f1',pred_f1, 'label_f1',label_f1)
            f1 += f1_score(target_list[0] , pred_target_list, average="macro") * lam + f1_score(target_list[1] , pred_target_list, average="macro") * (1 - lam)

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)

                train_f1 = f1 / args.log_interval
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || train f1-score {train_f1:4.4}"
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

        # scheduler.step()

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

            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            
            # val_f1
            if val_f1 > best_val_f1:
                print(f"New best model for val f1-score : {val_f1:4.4}! saving the best model..")
                torch.save(
                    model.module.state_dict(), f"{save_dir}/best.pth"
                )
                best_val_f1 = val_f1
                counter = 0
            else:
                counter +=1

            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print("-" * 100)
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || f1-score : {val_f1:4.4}["
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, best f1-score : {best_val_f1:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_figure("results", figure, epoch)
            print()
            # wandb.log({"acc": val_acc, "loss" : val_loss, "f1-score" : val_f1})
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os

    load_dotenv(verbose=True)
    model_name = 'Efficient_b4'
    batch_size = 64
    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=8, help="number of epochs to train (default: 8)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskSplitByProfileDataset",
        help="dataset augmentation type (default: MaskSplitByProfileDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs="+",
        type=list,
        default=[128, 96],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=batch_size,
        help="input batch size for training (default: {batch_size})",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default=f"{model_name}", help=f"model type (default: {model_name})"
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
        "--criterion",
        type=str,
        default="focal",
        help="criterion type (default: focal)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default=f"{model_name}_{batch_size}_", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
