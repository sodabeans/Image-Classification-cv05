import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

# f1
from sklearn.metrics import f1_score

# wandb
import wandb # reference: https://greeksharifa.github.io/references/2020/06/10/wandb-usage/#pytorch

# AutoML
import nni
from nni.utils import merge_parameter
# from pytorchtools import EarlyStopping

# grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

from adamp import AdamP, SGDP

# mtl
import torch
import torch.nn as nn
from fastai.vision import *
from model import MultiTaskModel


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
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

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
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


# mtl
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, age, gender, mask):

        crossEntropy = nn.CrossEntropyLoss()

        loss0 = crossEntropy(preds[0], age)
        loss1 = crossEntropy(preds[1],gender)
        loss2 = crossEntropy(preds[2],mask)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1+loss2


def train(data_dir, model_dir, args):
    patience = 10
    counter = 0

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    wandb.run.name = save_dir

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
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
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    wandb.watch(model)

    # -- criterion
    criterion = create_criterion(args.criterion)  # default: cross_entropy

    # --optimizer
    if args.optimizer == 'adamp':
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2, nesterov=True)
    elif args.optimizer == 'sgdp':
        optimizer = SGDP(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )

    # --scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    elif args.scheduler == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch,
                            last_epoch=-1, verbose=False)
    else:
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0

    

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        f1 = 0

        for idx, train_batch in enumerate(train_loader):
            if args.model == 'MultiTaskModel':
                inputs, (age_labels, gender_labels, mask_labels) = train_batch
                inputs = inputs.to(device)
                age_labels = age_labels.to(device)
                gender_labels = gender_labels.to(device)
                mask_labels = mask_labels.to(device)
                labels = MaskBaseDataset.encode_multi_class(mask_labels, gender_labels, age_labels)
                
                optimizer.zero_grad()

                age_outs, gender_outs, mask_outs = model(inputs)

                age_preds = torch.argmax(age_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                mask_preds = torch.argmax(mask_outs, dim=-1)
                preds = MaskBaseDataset.encode_multi_class(mask_preds, gender_preds, age_preds)

                age_loss = criterion(age_outs, age_labels)
                gender_loss = criterion(gender_outs, gender_labels)
                mask_loss = criterion(mask_outs, mask_labels)

                loss = age_loss + gender_loss + mask_loss
            else:
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()

            pred_f1 = torch.clone(preds).detach().cpu().numpy()
            label_f1 = torch.clone(labels).detach().cpu().numpy()
            f1 += f1_score(label_f1, pred_f1, average="macro")

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)

                train_f1 = f1 / args.log_interval

                print(
                    f"Epoch[{epoch + 1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || train f1-score {train_f1:4.4} "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1", train_f1, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                f1 = 0

        scheduler.step()

        # for wandb
        example_images = []
        grad_cams = []
        # start
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []
            figure = None

            for val_batch in val_loader:
                if args.model == 'MultiTaskModel':
                    inputs, (age_labels, gender_labels, mask_labels) = val_batch
                    age_labels = age_labels.to(device)
                    gender_labels = gender_labels.to(device)
                    mask_labels = mask_labels.to(device)
                    labels = MaskBaseDataset.encode_multi_class(mask_labels, gender_labels, age_labels)

                    age_outs, gender_outs, mask_outs = model(inputs)

                    age_loss = criterion(age_outs, age_labels)
                    gender_loss = criterion(gender_outs, gender_labels)
                    mask_loss = criterion(mask_outs, mask_labels)

                    loss_item = age_loss + gender_loss + mask_loss
                    loss_item = loss_item.cpu().numpy()

                    age_preds = torch.argmax(age_outs, dim=-1)
                    gender_preds = torch.argmax(gender_outs, dim=-1)
                    mask_preds = torch.argmax(mask_outs, dim=-1)
                    preds = MaskBaseDataset.encode_multi_class(mask_preds, gender_preds, age_preds)
                else:
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

                val_f1_items.append(f1_item)

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
                
                example_images.append(wandb.Image(
                    inputs[0], caption="Pred: {} Truth: {}".format(preds[0].item(), labels[0])
                ))

                # grad-cam
                def gradcam(inputs, model):
                    rgb_img = np.transpose(
                        np.float32(np.clip(inputs[0].cpu().numpy(), 0, 1)), (1, 2, 0)
                    )
                    target_layers = [model.module.fc()]
                    cam = GradCAM(
                        model=model, target_layers=target_layers, use_cuda=use_cuda
                    )
                # print(torch.unsqueeze(inputs[i], 0))
                    grayscale_cam = cam(input_tensor=torch.unsqueeze(inputs[0], 0))
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(
                        rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
                    )
                    return visualization

            # visualization = gradcam(inputs, model)
            # grad_cams.append(wandb.Image(
            #     visualization, caption="Pred: {} Truth: {}".format(preds[0].item(), labels[0]),
            # ))

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_loader)

            val_f1 = np.sum(val_f1_items) / len(val_loader)
            scheduler.step(val_f1)

            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)

            if val_f1 > best_val_f1:
                print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
                counter = 0
            else:
                counter += 1
            
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.4}, loss: {val_loss:4.2} || f1: {val_f1:4.2%} "
                f"best acc : {best_val_acc:4.4}, best loss: {best_val_loss:4.2}, best f1: {best_val_f1:4.2%}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_figure("results", figure, epoch)
            print()

            wandb.log({
                # "Img": example_images,
                # "Grad_CAM": grad_cams,
                "Val Loss": val_loss,
                "Val Acc": val_acc,
                "Val F1": val_f1,
                "figure": figure,
            })

            if counter > patience:
                print("Early Stopping...")
                break

            # early_stopping(val_loss, model)
        #end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    wandb.init(project="final", reinit=True) #TODO

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ResNet', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='adamp', help='optimizer type (default: adamp)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (list: cross_entropy, f1, focal, label_smoothing)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler decay step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--scheduler', default='default', help='cosine, lambda, default')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    # tuner_params = nni.get_next_parameter()
    # args = vars(merge_parameter(args(), tuner_params))
    print(args)

    wandb.config.update(args)
    wandb.run.save()

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)