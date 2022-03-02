from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

# Multi-Task Learning
# import fastai
# from fastai.vision import *
# from fastai.vision.learner import create_body, create_head

from dataset import MaskBaseDataset
# https://teamlab.github.io/jekyllDecent/blog/tutorials/fast.ai-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC-%EC%84%A4%EC%B9%98%ED%95%98%EA%B3%A0-CNN-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0
# https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class SqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.SqueezeNet = models.squeezenet1_0(pretrained=True)
        self.SqueezeNet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.SqueezeNet.num_classes = num_classes

    def forward(self, x):
        x = self.SqueezeNet(x)
        return x
# done

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.AlexNet = models.alexnet(pretrained=True)
        self.AlexNet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.AlexNet.num_classes = num_classes

    def forward(self, x):
        x = self.AlexNet(x)
        return x
# done


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ResNet = models.resnet18(pretrained=True)
        self.ResNet.fc = nn.Linear(512, num_classes)
        self.ResNet.num_classes = num_classes

    def forward(self, x):
        x = self.ResNet(x)
        return x
# done


class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.GoogLeNet = models.googlenet(pretrained=True)
        self.GoogLeNet.fc = nn.Linear(1024, num_classes)
        self.GoogLeNet.num_classes = num_classes

    def forward(self, x):
        x = self.GoogLeNet(x)
        return x
# done

"""
class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.DenseNet = models.densenet161(pretrained=True)
        self.DenseNet.classifier = nn.Linear(1000, num_classes)
        self.DenseNet.num_classes = num_classes

    def forward(self, x):
        x = self.DenseNet(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.VGG = models.vgg16(pretrained=True)
        self.VGG.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        self.VGG.num_classes = num_classes

    def forward(self, x):
        x = self.VGG(x)
        return x


class Inception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.Inception = models.inception_v3(pretrained=True)
        self.Inception.fc = nn.Linear(2048, num_classes)
        self.Inception.num_classes = num_classes

    def forward(self, x):
        x = self.Inception(x)
        return x


class ShuffleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ShuffleNet = models.shufflenet_v2_x1_0(pretrained=True)
        self.ShuffleNet.num_classes = num_classes

    def forward(self, x):
        x = self.ShuffleNet(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.MobileNetV2 = models.mobilenet_v2(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )
        self.MobileNetV2.num_classes = num_classes

    def forward(self, x):
        x = self.MobileNetV2(x)
        return x


class ResNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ResNeXt = models.resnext50_32x4d(pretrained=True)
        self.ResNeXt.num_classes = num_classes

    def forward(self, x):
        x = self.ResNeXt(x)
        return x


class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.WideResNet = models.wide_resnet50_2(pretrained=True)
        self.WideResNet.num_classes = num_classes

    def forward(self, x):
        x = self.WideResNet(x)
        return x


class MNASNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.MNASNet = models.mnasnet1_0(pretrained=True)
        self.MNASNet.num_classes = num_classes

    def forward(self, x):
        x = self.MNASNet(x)
        return x


class RegNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.RegNet = models.regnet_x_3_2gf(pretrained=True)
        self.RegNet.num_classes = num_classes

    def forward(self, x):
        x = self.RegNet(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.EfficientNet = models.efficientnet_b0(pretrained=True)
        # self.EfficientNet.fc = nn.Linear(128, num_classes)
        self.EfficientNet.num_classes = num_classes

    def forward(self, x):
        x = self.EfficientNet(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128, num_classes, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        return x
"""


class MultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.AgeModel = models.googlenet(pretrained=True)
        self.AgeModel.fc = nn.Linear(1024, 3)

        self.GenderModel = models.resnet18(pretrained=True)
        self.GenderModel.fc = nn.Linear(512, 2)

        self.MaskModel = models.resnet18(pretrained=True)
        self.MaskModel.fc = nn.Linear(512, 3)


    def forward(self, x):
        age = self.AgeModel(x)
        gender = self.GenderModel(x)
        mask = self.MaskModel(x)
        return age, gender, mask

