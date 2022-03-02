import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda: print("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()
    
    if model.training:
        print("validation state but training...")
    else:
        print("not training...")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    age_1 = 0
    age_2 = 0
    age_3 = 0
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            if args.model == 'MultiTaskModel':
                age_outs, gender_outs, mask_outs = model(images)
                age_preds = torch.argmax(age_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                mask_preds = torch.argmax(mask_outs, dim=-1)
                pred = MaskBaseDataset.encode_multi_class(mask_preds, gender_preds, age_preds)
            else:
                pred = model(images)
                pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
            if pred == 2 or pred == 5 or pred == 8 or pred == 11 or pred == 14 or pred == 17:
                age_3 = age_3 + 1
            elif pred == 1 or pred == 4 or pred == 7 or pred == 10 or pred == 13 or pred == 16:
                age_2 = age_2 + 1
            else:
                age_1 = age_1 + 1
    
    print(age_1)
    print(age_2)
    print(age_3)

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
