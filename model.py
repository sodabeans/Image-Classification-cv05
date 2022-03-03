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


class Res18(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        self.drop_rate = dropout
        self.res18 = timm.create_model(
            "resnet18",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=self.drop_rate,
        )

    def target_layer(self):
        return self.res18.layer4

    def forward(self, x):
        x = self.res18(x)
        return x


class Res50(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        self.drop_rate = dropout
        self.res50 = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=self.drop_rate,
        )

    def target_layer(self):
        return self.res50.layer4

    def forward(self, x):
        x = self.res50(x)
        return x


class Efficient(nn.Module):
    def __init__(self, num_classes, dropout=None):
        super().__init__()
        self.drop_rate = dropout
        self.eff_b4 = timm.create_model(
            "tf_efficientnet_b4",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=self.drop_rate,
        )  # tf_efficientnet_b4

    def target_layer(self):
        return self.eff_b4.conv_head

    def forward(self, x):
        x = self.eff_b4(x)
        return x


# nf_regnet_b1


class NFRegnet(nn.Module):
    def __init__(self, num_classes, dropout=None):
        super().__init__()
        self.drop_rate = dropout
        self.nfr = timm.create_model(
            "nf_regnet_b1",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=self.drop_rate,
        )

    def target_layer(self):
        return self.nfr.final_conv

    def forward(self, x):
        x = self.nfr(x)
        return x


# mixnet_m
class Mixnet(nn.Module):
    def __init__(self, num_classes, dropout=None):
        super().__init__()
        self.drop_rate = dropout
        self.mix = timm.create_model(
            "mixnet_m",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=self.drop_rate,
        )

    def target_layer(self):
        return self.mix.conv_head

    def forward(self, x):
        x = self.mix(x)
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
