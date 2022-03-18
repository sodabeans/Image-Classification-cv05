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

import torch
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from model import *
from glob import glob

import random


def load_model(saved_model, num_classes, device):
    model_cls = Res18Model
    model = model_cls(num_classes=num_classes)

    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


if __name__ == "__main__":
    model_dir = "./model_save"
    num_classes = 18

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, num_classes, device=device).to(device)
    model.eval()

    image_paths = glob("/ml/input/data/eval/images/*")
    image_paths = [file for file in image_paths if not file.startswith(".")]

    i = random.randint(0, len(image_paths) - 1)

    # target_layers = [model.res18.layer4[-1]]
    target_layers = [model.target_layer()]
    
    img = cv2.imread(image_paths[i], 1)[:, :, ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = np.float32(img) / 255

    input_tensor = torch.permute(
        torch.unsqueeze(torch.Tensor(rgb_img).to(device), 0), (0, 3, 1, 2)
    )

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
    )
    visualization_save = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    plt.matshow(
        visualization
    )  # https://geniekj.blogspot.com/2019/11/vscode-python.html
    plt.show(block=False)