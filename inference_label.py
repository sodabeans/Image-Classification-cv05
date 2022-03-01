import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def decode_multi_label(multi_class_label):
    # print(multi_class_label)
    # multi_class_label = multi_class_label

    mask_label = multi_class_label[:3]
    # print(mask_label)
    gender_label = multi_class_label[3:5]
    age_label = multi_class_label[5:]
    return (
        np.argmax(mask_label) * 6 + np.argmax(gender_label) * 3 + np.argmax(age_label)
    )


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_labels  # 8
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    # print(args.resize)
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # https://coder-question-ko.com/cq-ko-blog/64450  # https://github.com/pytorch/pytorch/issues/5301
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            # print(pred)
            pred = model(images)
            pred = torch.sigmoid(pred).data > 0.5
            pred = pred.long()
            pred = np.apply_along_axis(
                decode_multi_label, 1, pred.cpu().numpy()
            )  # np.apply_along_axis(decode_multi_label,1,a)
            preds.extend(pred)

    info["ans"] = preds
    num = model_dir.split("/")[-1]
    info.to_csv(os.path.join(output_dir, f"output_{num}.csv"), index=False)
    print(f"Inference Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=300,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=(512, 384),
        help="resize size for image when you trained (default: (512, 384))",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Res18Model",
        help="model type (default: Res18Model)",
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_CHANNEL_MODEL", "./model")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
