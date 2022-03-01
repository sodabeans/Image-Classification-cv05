import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


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


class Res18Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.res18 = models.resnet18(pretrained=True)
        self.res18.fc = nn.Sequential(
            # nn.Linear(2048, 1024),
            #  nn.Linear(1024, 256),
            nn.Linear(512, num_classes)
        )

    def target_layer(self):
        return self.res18.layer4

    def forward(self, x):
        x = self.res18(x)
        return x


class Efficient(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eff_b4 = timm.create_model(
            "tf_efficientnet_b4",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.7,
        )  # tf_efficientnet_b4

    def target_layer(self):
        return self.eff_b4.conv_head

    def forward(self, x):
        x = self.eff_b4(x)
        return x


# nf_regnet_b1


class NFRegnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.nfr = timm.create_model("nf_regnet_b1", pretrained=True)
        self.nfr.head.fc = nn.Sequential(
            nn.Linear(960, 256),
            #  nn.Linear(1024, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.nfr(x)
        return x


# deit_base_distilled_patch16_384
class Deit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.deit = timm.create_model(
            "deit_base_distilled_patch16_384", pretrained=True
        )
        self.deit.head = nn.Sequential(
            nn.Linear(768, 256),
            #  nn.Linear(1024, 256),
            nn.Linear(256, num_classes),
        )
        self.deit.head_dist = nn.Sequential(
            nn.Linear(768, 256),
            #  nn.Linear(1024, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.deit(x)
        return torch.stack(list(x), dim=0)


class Swin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        self.swin.head = nn.Linear(1024, num_classes)

    def target_layer(self):
        return self.swin.avgpool

    def forward(self, x):
        x = self.swin(x)
        return x


class VIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_384", pretrained=True)
        self.vit.head = nn.Sequential(
            nn.Linear(768, 256),
            #  nn.Linear(1024, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.vit(x)
        return x


# n tresnet_l
# class TResnet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.tres = timm.create_model("tresnet_l", pretrained=True)
#         self.tres.head.fc = nn.Sequential(
#             nn.Linear(2432, 512),
#             #  nn.Linear(1024, 256),
#             nn.Linear(512, num_classes),
#         )

#     def forward(self, x):
#         x = self.tres(x)
#         return x


# # densenet121
# class Dense(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.dense = timm.create_model("densenet121", pretrained=True)
#         self.dense.classifier = nn.Sequential(
#             # nn.Linear(2432, 512),
#             nn.Linear(1024, 256),
#             nn.Linear(256, num_classes),
#         )

#     def forward(self, x):
#         x = self.dense(x)
#         return x


# class InceptionV3Model(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.inception = models.inception_v3(pretrained=True)
#         self.inception.fc = nn.Sequential(
#             nn.Linear(2048, 1024), nn.Linear(1024, 256), nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.inception(x)
#         return x[0]
