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
import yaml


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
from sklearn.model_selection import StratifiedKFold


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
        dirs = glob.glob(f"{path}*")
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ]  # 여기 잘 모르겠음. match하는 것을 찾는 것 같음
        i = [int(m.groups()[0]) for m in matches if m]  # match한게 있다면 거기서 하나 뽑아서 저렇게 하면
        n = max(i)
        return f"{path}{n}"
    else:
        dirs = glob.glob(f"{path}*")
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ]  # 여기 잘 모르겠음. match하는 것을 찾는 것 같음
        i = [int(m.groups()[0]) for m in matches if m]  # match한게 있다면 거기서 하나 뽑아서 저렇게 하면
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers, sampler=None):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset, indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set = torch.utils.data.Subset(dataset, indices=valid_idx)

    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True,
        shuffle=False,
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,
    )

    # 생성한 DataLoader 반환
    return train_loader, val_loader


# grad-cam
def gradcam(inputs, i, model):
    rgb_img = np.transpose(
        np.float32(np.clip(inputs[i].cpu().numpy(), 0, 1)), (1, 2, 0)
    )
    target_layers = [model.module.target_layer()]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # print(torch.unsqueeze(inputs[i], 0))
    grayscale_cam = cam(input_tensor=torch.unsqueeze(inputs[i], 0))
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
    )
    return visualization


# 결국은 model을 저장해주는 path를 만들기 위한 코드임


def train(data_dir, model_dir, args, sweep_id):
    seed_everything(args.seed)

    if args.path == "yes":
        save_dir = increment_path(os.path.join(model_dir, args.name))
    else:
        save_dir = increment_path(os.path.join(model_dir, args.name), exist_ok=True)

    # -- settings
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_module = getattr(
        import_module("dataset"), args.dataset,
    )  # default: MaskBaseDataset
    dataset = dataset_module(data_dir=data_dir, target_type=args.target_type,)
    num_classes = dataset.num_classes  # 18

    # transform
    transform = BaseTransform(
        resize=args.resize, mean=dataset.mean, std=dataset.std, cutmix=args.cutmix,
    )
    dataset.set_transform(transform)

    # -- augmentation
    if args.augmentation == "CustomAugmentation":
        augmentation_module = getattr(
            import_module("dataset"), args.augmentation
        )  # default: BaseAugmentation
        augmentation = augmentation_module()

    # data split
    train_set, val_set = dataset.split_dataset()  # 저 dataset내에 이게 있어서 쓸 수 있음.

    # print(len(train_set))
    # -- sampler
    # if arg.sampler == "Weight":
    #     class_sample_counts = [
    #         2745,
    #         2050,
    #         415,
    #         3660,
    #         4085,
    #         545,
    #         549,
    #         410,
    #         83,
    #         732,
    #         817,
    #         109,
    #         549,
    #         410,
    #         83,
    #         732,
    #         817,
    #         109,
    #     ]
    #     weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)
    #     samples_weights = [weights[t[1]] for t in train_set]
    #     # https://discuss.pytorch.org/t/how-to-augment-the-minority-class-only-in-an-unbalanced-dataset/13797/3
    #     # print(samples_weights)
    #     # print(weights)
    #     # print()
    #     sampler = WeightedRandomSampler(
    #         weights=samples_weights, num_samples=len(samples_weights), replacement=True
    #     )  # 그러면 dataloader에서 어떤 샘플들을 뽑을 때 이 각 클래스 확률로 지정된 확률 내에서 뽑힌다는 거구만,
    #     # 이거 쓰면 불균형 문제 해소에 도움이 되겠다.(각 배치당 불균형해소 될 듯!)
    # elif arg.sampler == "Imbalance":  # Imbalance sampler
    #     sampler = ImbalancedDatasetSampler(train_set)

    # -- logging -> 튜닝을 하게 되면 각 폴드마다?는 아님. 흐음 그럼 여기 안들어가는게 맞긴 하겠네

    sweep_dir = os.path.join(save_dir, f"{sweep_id}")

    logger = SummaryWriter(log_dir=sweep_dir)
    with open(os.path.join(sweep_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            vars(args), f, ensure_ascii=False, indent=4
        )  # 특정 json파일에 이러한 config를 적어준다.

    # kfold
    labels = [
        dataset.encode_multi_class(mask, gender, age)
        for mask, gender, age in zip(
            dataset.mask_labels, dataset.gender_labels, dataset.age_labels
        )
    ]

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)
    kfold_val_loss = []
    kfold_val_acc = []
    kfold_val_f1 = []

    # Stratified KFold를 사용해 Train, Valid fold의 Index를 생성합니다.
    # labels 변수에 담긴 클래스를 기준으로 Stratify를 진행합니다.
    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(dataset.image_paths, labels)
    ):
        # print(len(train_idx))
        # print(len(valid_idx))
        # print(train_idx)
        fold += 1

        fold_dir = os.path.join(sweep_dir, f"Fold_{fold}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        wandb.init(
            project="Image_Classification",
            group=save_dir,
            job_type=sweep_id,
            # reinit=True,
            config=vars(args),
            name=f"{save_dir}/ {sweep_id} / Fold_{fold}",
            resume="allow",
        )

        print(
            "***************************************************************************************************************"
        )
        print(f"                                          Fold [{fold}/{n_splits}]")
        print(
            "***************************************************************************************************************"
        )
        # 생성한 Train, Valid Index를 getDataloader 함수에 전달해 train/valid DataLoader를 생성합니다.
        # 생성한 train, valid DataLoader로 이전과 같이 모델 학습을 진행합니다.
        train_loader, val_loader = getDataloader(
            dataset,
            train_idx,
            valid_idx,
            batch_size=args.batch_size,
            num_workers=0,
            # sampler=sampler,
        )

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(num_classes=num_classes, dropout=args.dropout).to(device)
        model = torch.nn.DataParallel(model)

        # wandb
        wandb.watch(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(
            import_module("torch.optim"), args.optimizer
        )  # default: SGD

        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-5,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=args.decay_step,
            threshold=0.001,
            verbose=True,
            min_lr=1e-6,
            threshold_mode="abs",
        )

        best_val_acc = 0
        best_val_f1 = 0
        best_val_loss = np.inf

        # early stopping
        patience = 5
        counter = 0

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

                if np.random.random() > 0.5 and args.cutmix == "yes":
                    inputs, labels = CutmixFace(0.8)(inputs, labels)

                # Albutmentation
                # inputs_np = (
                #     torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                # ).astype(np.int)
                # inputs_np = inputs_np[..., ::-1].copy()

                # custom augmentation
                if args.augmentation == "CustomAugmentation":
                    aug_inputs = []
                    for input, label in zip(inputs, labels):  # inputs_np
                        if label not in [0, 1, 3, 4]:
                            aug_input = augmentation(input)
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
                f1 += f1_score(label_f1, pred_f1, average="macro")
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)

                    train_f1 = f1 / args.log_interval

                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || train f1-score {train_f1:4.4}"
                    )

                    loss_value = 0
                    matches = 0
                    f1 = 0

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                # example_imgaes = []
                # grad_cams = []
                val_loss_items = []
                val_acc_items = []
                val_f1_items = []
                # wandb

                figure = None
                for val_idx, val_batch in enumerate(val_loader):
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

                    # if val_idx % 40 == 0:

                    #     # randint -> random image show in wandb
                    #     i = random.randint(0, len(inputs) - 1)

                    #     # wandb
                    #     example_imgaes.append(
                    #         wandb.Image(
                    #             inputs[i],
                    #             caption="Pred: {} Truth: {}".format(
                    #                 preds[i].item(), labels[i]
                    #             ),
                    #         )
                    #     )

                    #     visualization = gradcam(inputs, i, model)

                    #     grad_cams.append(
                    #         wandb.Image(
                    #             visualization,
                    #             caption="Pred: {} Truth: {}".format(
                    #                 preds[i].item(), labels[i]
                    #             ),
                    #         )
                    #     )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                val_f1 = np.sum(val_f1_items) / len(val_loader)
                scheduler.step(val_f1)  # lr steping

                # 여기서 모델을 계속 업데이트되면서 저장되어있어야하는게 맞는 것 같고,
                # 이 best score로 kfold내는게 맞는 것 같다.
                best_val_loss = min(best_val_loss, val_loss)
                best_val_acc = max(best_val_acc, val_acc)

                if (
                    val_f1 > best_val_f1
                ):  # 여기서 일종의 threshold를 정해주자. 한 0.005정도는 넘어야 new로 업데이트하게
                    print(
                        f"New best model for val f1-score : {val_f1:4.4}! saving the best model.."
                    )
                    torch.save(
                        model.module.state_dict(), f"{fold_dir}/best.pth"
                    )  # best 모델이 계속 업데이트될 듯
                    best_val_f1 = val_f1
                    counter = 0
                    print(
                        "***************************************************************************************************************"
                    )
                    print(
                        f"Fold_{fold} [Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} , F1-score : {best_val_f1:4.4} "
                    )
                    print(
                        "***************************************************************************************************************"
                    )
                else:
                    counter += 1
                if counter > patience:
                    print("Early Stopping...")
                    break
                torch.save(model.module.state_dict(), f"{fold_dir}/last.pth")

                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} , F1-score : {val_f1:4.4}|| "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, Best F1-score : {best_val_f1:4.4}"
                )

                wandb.log(
                    {
                        # "Examples": example_imgaes,
                        # "Grad_CAM": grad_cams,
                        "Valid Acc": val_acc,
                        "Valid Loss": val_loss,
                        "Valid F1-score": val_f1,
                    }
                )

        kfold_val_loss.append(best_val_loss)
        kfold_val_acc.append(best_val_acc)
        kfold_val_f1.append(best_val_f1)

        # 약간 트릭을 써야할 듯, mean_acc는 그냥 저장하고 figure를 logging으로 보내는거.
        # 아니면 그냥 mean만 tracking하도록 해야함 이래도 되긴 함.
        # epoch마다 각 fold의 그거를 저장해주는? 그 logging이 필요한데, 그렇게 되더라도
        # 이거는 sweep에서 여러 fold가 나뉘는게 아니라 한 정보가 됨.
        # https://colab.research.google.com/drive/181GCGp36_75C2zm7WLxr9U2QjMXXoibt
        # https://www.kaggle.com/ayuraj/efficientnet-mixup-k-fold-using-tf-and-wandb#%F0%9F%9A%8B-Train-with-W&B

        if fold == 5:
            mean_acc = np.sum(kfold_val_acc) / n_splits
            mean_loss = np.sum(kfold_val_loss) / n_splits
            mean_f1 = np.sum(kfold_val_f1) / n_splits
            wandb.log(
                {
                    "Mean_Acc": mean_acc,
                    "Mean_Loss": mean_loss,
                    "Mean_F1_Score": mean_f1,
                }
            )
        wandb.join()
        wandb.finish()

    # 애내는 sweep이랑 관련이 있게 될 것 같다.

    print(
        "***************************************************************************************************************"
    )
    print(
        f"Fold_{fold} mean acc : {mean_acc:4.2%}, mean loss: {mean_loss:4.2} , mean F1-score : {mean_f1:4.4}"
    )
    print(
        "***************************************************************************************************************"
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

    ############################### sweep default 로 argparser처럼 동작하기 때문에 argparser를 이용해줘야한다.

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
        help="data augmentation type (default: CustomAugmentation)",
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
        default=300,
        help="input batch size for validing (default: 300)",  # https://study-grow.tistory.com/entry/RuntimeError-cuDNN-error-CUDNNSTATUSNOTSUPPORTED-This-error-may-appear-if-you-passed-in-a-non-contiguous-input-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0%EB%B2%95
    )  # 이거 줄이니까 문제 해결
    parser.add_argument(
        "--model", type=str, default="Res18", help="model type (default: Res18)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help=" drop rate (default: 0.5)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer type (default: Adam)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="learning rate (default: 1e-3)"
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
        "--decay_step",
        type=int,
        default=2,
        help="learning rate scheduler deacy step (default: 2)",
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

    parser.add_argument(
        "--path", default="no", help="direct cmd run only have new exp path"
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

    # with open("config_default.yaml") as f:
    #     config_defaults = yaml.safe_load(f)

    with open("train.yaml") as f:
        sweep_config = yaml.safe_load(f)  # sweep config.yaml
    sweep_id = wandb.sweep(sweep_config, project="Image_Classification")

    # config_defaults = config_defaults["parameters"]
    data_dir = args.data_dir
    model_dir = args.model_dir

    # train(data_dir, model_dir, config=config_defaults)
    # wandb.config.update(args)
    wandb.agent(
        sweep_id,
        function=train(data_dir, model_dir, args=args, sweep_id=sweep_id),
        count=10,
    )
    # sweep이 결국 별게 아니라 저 새로운 config로 저 파일을 다시 돌리는거야. 이 파일을 이 파일을 다시 돌렸을 때 sweepid별로 나뉘어지려면..?

    # sweep 사용시 cuda oom 에러 자꾸 발생하는거보면 내 코드상의 문제가 있음 이거 체크해야
    # 이거 해결하는게 이번 목표

