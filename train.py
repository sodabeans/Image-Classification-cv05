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

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import MaskBaseDataset
from loss import create_criterion
from sklearn.metrics import f1_score


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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    wandb.run.name = save_dir
    wandb.run.save()

    # -- settings
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
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
        args.dataset,  # iomport module로 현재 디렉토리의 dataset모듈을 불러오는거 ㅇㅇ 거기의 BaseAugmenation을 가져오는거
    )  # default: MaskBaseDataset
    dataset = dataset_module(data_dir=data_dir,)
    num_classes = dataset.num_classes  # 18

    # transform
    transform = BaseTransform(resize=args.resize, mean=dataset.mean, std=dataset.std,)
    dataset.set_transform(transform)

    # -- augmentation
    augmentation_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    augmentation = augmentation_module()

    # dataset.set_augmentation(augmentation)

    # data split
    train_set, val_set = dataset.split_dataset()  # 저 dataset내에 이게 있어서 쓸 수 있음.

    # print(len(train_set))
    # -- sampler
    if args.sampler == "Weight":
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
        weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)
        samples_weights = [t[1] for t in train_set]
        # https://discuss.pytorch.org/t/how-to-augment-the-minority-class-only-in-an-unbalanced-dataset/13797/3
        # print(samples_weights)
        # print(weights)
        # print()
        sampler = WeightedRandomSampler(
            weights=samples_weights, num_samples=len(samples_weights), replacement=True
        )  # 그러면 dataloader에서 어떤 샘플들을 뽑을 때 이 각 클래스 확률로 지정된 확률 내에서 뽑힌다는 거구만,
        # 이거 쓰면 불균형 문제 해소에 도움이 되겠다.(각 배치당 불균형해소 될 듯!)
    elif args.sampler == "Imbalance":  # Imbalance sampler
        sampler = ImbalancedDatasetSampler(train_set)

    # -- data_loader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        # num_workers=multiprocessing.cpu_count() // 2,
        # shuffle=True,  # https://stackoverflow.com/questions/61033726/valueerror-sampler-option-is-mutually-exclusive-with-shuffle-pytorch
        # pin_memory=use_cuda,
        drop_last=True,
    )  # dataset클래스니까, 이걸 이렇게 loader로 만들어줄 수 있음.

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        # num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        # pin_memory=use_cuda,
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
    # https://jonhyuk0922.tistory.com/162
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
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

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            vars(args), f, ensure_ascii=False, indent=4
        )  # 특정 gson파일에 이러한 config를 적어준다.

    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf

    # early stopping
    patience = 3
    counter = 0

    # print([list(model.children())[0]])
    # print([list(list(list(model.children())[0].children())[0].children())[-2]])

    for epoch in range(args.epochs):
        # train loop
        epoch += 1
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

            # https://discuss.pytorch.org/t/how-to-augment-the-minority-class-only-in-an-unbalanced-dataset/13797/2
            if np.random.random() > 0.5:
                inputs, labels = CutmixFace(0.8)(inputs, labels)

            # custom augmentation
            aug_inputs = []
            for input, label in zip(inputs, labels):
                if label not in [0, 1, 3, 4]:
                    aug_input = CustomAugmentation()(input)
                    aug_inputs.append(aug_input)
                else:
                    aug_inputs.append(input)
            inputs = torch.stack(aug_inputs)

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
            # print(f1_score(label_f1, pred_f1, average="macro"))
            f1 += f1_score(label_f1, pred_f1, average="macro")
            if (
                idx + 1
            ) % args.log_interval == 0:  # 적당 수준 돌았을 때 배치가 어느정도 우리가 설정한거에 맞게 돌면 찍는거 이 기준으로 평균도
                train_loss = (
                    loss_value / args.log_interval
                )  # 일종의 평균 로스 ( 로스를 다 찍는게 아닌 배치 평균찍는거-> 좀 더 생각)
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)  # lr scheduler를 쓰기 때문에 사용.

                train_f1 = f1 / args.log_interval

                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "  # 총 몇 에폭중 몇 에폭째인지 총 배치 중에 몇번쨰 배치인지
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || train f1-score {train_f1:4.4}"
                )
                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )  # 텐서보드에 넣어주는거.
                logger.add_scalar(
                    "Train/F1-score", train_f1, epoch * len(train_loader) + idx
                )

                loss_value = 0
                matches = 0
                f1 = 0

        # val loop

        print("Calculating validation results...")
        model.eval()
        val_loss_items = []
        val_acc_items = []
        val_f1_items = []
        # wandb
        example_imgaes = []
        grad_cams = []
        figure = None
        for val_batch in val_loader:
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            if np.random.random() > 0.5:
                inputs, labels = CutmixFace(0.8)(inputs, labels)

            # custom augmentation
            # https://discuss.pytorch.org/t/how-to-augment-the-minority-class-only-in-an-unbalanced-dataset/13797/2
            aug_inputs = []
            for input, label in zip(inputs, labels):
                if label not in [0, 1, 3, 4]:
                    aug_input = CustomAugmentation()(input)
                    aug_inputs.append(aug_input)
                else:
                    aug_inputs.append(input)
            inputs = torch.stack(aug_inputs)

            with torch.no_grad():
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

            # randint -> random image show in wandb
            i = random.randint(0, len(inputs) - 1)

            # wandb
            example_imgaes.append(
                wandb.Image(
                    inputs[i],
                    caption="Pred: {} Truth: {}".format(preds[i].item(), labels[i]),
                )
            )

            # grad-cam
            def gradcam(inputs, i, model):
                rgb_img = np.transpose(
                    np.float32(np.clip(inputs[i].cpu().numpy(), 0, 1)), (1, 2, 0)
                )
                target_layers = [model.module.target_layer()]
                cam = GradCAM(
                    model=model, target_layers=target_layers, use_cuda=use_cuda
                )
                # print(torch.unsqueeze(inputs[i], 0))
                grayscale_cam = cam(input_tensor=torch.unsqueeze(inputs[i], 0))
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(
                    rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
                )
                return visualization

            visualization = gradcam(inputs, i, model)

            # rgb_img = np.transpose(
            #     np.float32(np.clip(inputs[i].cpu().numpy(), 0, 1)), (1, 2, 0)
            # )
            # target_layers = [
            #     list(list(list(model.children())[0].children())[0].children())[-3] # 이 숫자만 바뀌면 되고, 이건 모델 종속적 colormap도 바꿀 수 있긴한데 굳이?>
            # ] # 이걸 애초에 모델에서 지정해서 꺼낼 수 있게!
            # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
            # # print(torch.unsqueeze(inputs[i], 0))
            # grayscale_cam = cam(input_tensor=torch.unsqueeze(inputs[i], 0))
            # grayscale_cam = grayscale_cam[0, :]
            # visualization = show_cam_on_image(
            #     rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
            # )
            grad_cams.append(
                wandb.Image(
                    visualization,
                    caption="Pred: {} Truth: {}".format(preds[i].item(), labels[i]),
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
        scheduler.step(val_f1)  # lr steping
        best_val_loss = min(best_val_loss, val_loss)
        # if val_acc > best_val_acc:
        #     print(
        #         f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
        #     )
        #     torch.save(
        #         model.module.state_dict(), f"{save_dir}/best.pth"
        #     )  # best 모델이 계속 업데이트될 듯
        #     best_val_acc = val_acc

        if val_f1 > best_val_f1:  # 여기서 일종의 treshold를 정해주자. 한 0.005정도는 넘어야 new로 업데이트하게
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
        if counter > patience:
            print("Early Stopping...")
            break
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
        logger.add_scalar("Val/F1-score", val_f1, epoch)
        logger.add_figure(
            "results", figure, epoch
        )  # 이렇게 figure를 우리가 원하는거를 tensorboard에 추가할 수도 있구나
        # wandb
        wandb.log(
            {
                "Examples": example_imgaes,
                "Grad_CAM": grad_cams,
                "Valid Acc": val_acc,
                "Valid Loss": val_loss,
                "Valid F1-score": val_f1,
            }
        )
        print()


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

    # wandb
    wandb.init(project="Image_Classification", reinit=True)

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
        default="CustomAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
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
        default=300,
        help="input batch size for validing (default: 300)",  # https://study-grow.tistory.com/entry/RuntimeError-cuDNN-error-CUDNNSTATUSNOTSUPPORTED-This-error-may-appear-if-you-passed-in-a-non-contiguous-input-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0%EB%B2%95
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
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
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

    # wandb

    # wandb.run.name = 'args.name'
    wandb.config.update(args)

    # wandb.run.save()

    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
