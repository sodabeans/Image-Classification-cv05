import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *
import torch.nn.functional as F

# import cv2

import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseTransform:
    def __init__(self, resize, mean, std, cutmix, **args):
        self.crop = CenterCrop((400, 300))
        self.cutmix = cutmix
        self.transform = transforms.Compose(
            [
                # 이거 대신 cutmix
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        if self.cutmix == "no":
            image = self.crop(image)
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CustomAugmentation:
    def __init__(self):  # resize, mean, std, **args):
        self.transform = RandomChoice(
            [
                # RandomBrightnessContrast(brightness_limit = (-0.1,0.1), contrast_limit=(-0.1,0.1),p=0.5),
                ColorJitter(
                    brightness=(0.2, 2),
                    contrast=(0.3, 2),
                    saturation=(0.2, 2),
                    hue=(-0.3, 0.3),
                ),  # ColorJitter(0.1, 0.1, 0.1, 0.1),  # 이건 의미가 있을 것 같음.
                RandomPerspective(),
                RandomRotation(
                    90
                ),  # https://stackoverflow.com/questions/60205829/pytorch-transforms-randomrotation-does-not-work-on-google-colab
                RandomAffine(degrees=(30, 70)),  #
                # RandomPosterize(2),
                RandomSolarize(192),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                # AddGaussianNoise(),
            ]
        )

        # m A.Compose(
        #     [
        #         # brightness, contrast, saturation을 무작위로 변경합니다.
        #         # brightness, contrast, saturation을 무작위로 변경합니다.
        #         A.ColorJitter(
        #             brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4
        #         ),
        #         # transforms 중 하나를 선택해 적용합니다.
        #         A.OneOf(
        #             [
        #                 # shift, scale, rotate 를 무작위로 적용합니다.
        #                 A.ShiftScaleRotate(rotate_limit=20, p=0.5,),
        #                 # affine 변환
        #                 # A.IAAAffine(shear=15, p=0.5, mode="constant"),
        #             ],
        #             p=1.0,
        #         ),
        #         # 수평 뒤집기
        #         A.HorizontalFlip(p=0.5),
        #         # blur
        #         A.Blur(p=0.1),
        #         # Contrast Limited Adaptive Histogram Equalization 적용
        #         A.CLAHE(p=0.1),
        #         # 각 채널의 bit 감소
        #         A.Posterize(p=0.1),
        #         # grayscale로 변환
        #         A.ToGray(p=0.1),
        #         # 무작위로 channel을 섞기
        #         A.ChannelShuffle(p=0.05),
        #         # A.ColorJitter(
        #         #     brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4
        #         # ),
        #         # # transforms 중 하나를 선택해 적용합니다.
        #         # A.OneOf(
        #         #     [
        #         #         # shift, scale, rotate 를 무작위로 적용합니다.
        #         #         A.ShiftScaleRotate(rotate_limit=20, p=0.5),
        #         #         # affine 변환
        #         #         A.Affine(shear=15, p=0.5),
        #         #         A.RandomRotate90(p=0.5),
        #         #     ],
        #         #     p=0.5,
        #         # ),
        #         # # 수평 뒤집기
        #         # A.OneOf(
        #         #     [
        #         #         A.Blur(blur_limit=(30, 30), p=0.5),
        #         #         # A.GaussianBlur(p=0.5),
        #         #         # A.MotionBlur(p=0.5),
        #         #         # A.OpticalDistortion(p=0.5),
        #         #     ],
        #         #     p=0.5,
        #         # ),
        #         # A.OneOf(
        #         #     [
        #         #         A.ToGray(p=0.1),
        #         #         A.ChannelShuffle(p=0.1),
        #         #         A.InvertImg(p=0.1),
        #         #         A.Solarize(p=0.1),
        #         #         A.Posterize(p=0.1),
        #         #         A.CLAHE(p=0.1),
        #         #     ],
        #         #     p=0.5,
        #         # ),
        #         # A.OneOf(
        #         #     [
        #         #         A.GaussNoise(p=0.5),
        #         #         A.MultiplicativeNoise(
        #         #             multiplier=[0.5, 1.5],
        #         #             elementwise=True,
        #         #             per_channel=True,
        #         #             p=0.5,
        #         #         ),
        #         #     ],
        #         #     p=0.5,
        #         # ),
        #         # A.HorizontalFlip(p=0.5),
        #         # A.VerticalFlip(p=0.5),
        #         # A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5),
        #         # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        #         # # Contrast Limited Adaptive Histogram Equalization 적용
        #         # # A.CLAHE(p=0.1),
        #         # A.OneOf(
        #         #     [
        #         #         # A.CoarseDropout(
        #         #         #     max_holes=8, max_height=20, max_width=20, p=0.5
        #         #         # ),
        #         #         A.ChannelDropout(p=0.5),
        #         #         A.Cutout(
        #         #             num_holes=50,
        #         #             max_h_size=40,
        #         #             max_w_size=40,
        #         #             fill_value=128,
        #         #             p=0.5,
        #         #         ),
        #         #     ],
        #         #     p=0.5,
        #         # ),
        #         # A.JpegCompression(p=0.1),
        #         ToTensorV2(),
        #     ]
        # )
        # transforms.Compose(
        #     [
        # CenterCrop((320, 256)),
        # Resize(resize, Image.BILINEAR),
        # ToTensor(),
        # Normalize(mean=mean, std=std),

    # for train # https://github.com/albumentations-team/albumentations_examples
    # https://www.facebook.com/groups/PyTorchKR/posts/1739555296184144/

    def __call__(self, image):
        # augmentation = self.transform(image=image)
        # return augmentation["image"]
        return self.transform(image)


class CutmixFace(object):  # 정답 설정도 필요할 듯. # 정답 설정도 뒤빠뀌어ㅑ함 저거에 맞게
    def __init__(self, ratio):  # 0.8
        self.ratio = ratio

    def __call__(self, input, labels):
        # print(input.shape)
        W = input.size()[3]
        H = input.size()[2]
        cx = W // 2
        cy = H // 2

        newH = round(H * self.ratio)
        newW = round(W * self.ratio)

        # input = CenterCrop((newH, newW))(input)
        # print(type(input))

        # print(input.shape)
        # print(input.size())
        # print(input.size()[0])
        rand_index = torch.randperm(input.size()[0])

        bbx1, bby1, bbx2, bby2 = (
            cx - (newW // 2),
            cy - (newH // 2),
            cx + (newW // 2),
            cy + (newH // 2),
        )
        input[:, :, bby1:bby2, bbx1:bbx2] = input[rand_index, :, bby1:bby2, bbx1:bbx2]
        labels = labels[rand_index]
        # print(input.shape)

        return input, labels

        # return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return "CutMix"


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3
    num_labels = 3 + 2 + 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_dir,
        target_type,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.target_type = target_type

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    # def set_augmentation(self, augmentation):
    #     self.augmentation = augmentation

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)

        if self.target_type == "multiclass":
            multi_class_label = self.encode_multi_class(
                mask_label, gender_label, age_label
            )

        elif self.target_type == "multilabel":
            multi_class_label = self.encode_multi_label(
                mask_label, gender_label, age_label
            )  # class수가 8

        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_label(mask_label, gender_label, age_label) -> int:
        mask_onehot = np.eye(3)[mask_label]
        gender_onehot = np.eye(2)[gender_label]
        age_onehot = np.eye(3)[age_label]
        return np.concatenate((mask_onehot, gender_onehot, age_onehot))

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    # @staticmethod
    # def decode_multi_class(multi_class_label):
    #     mask_label = multi_class_label[:3]
    #     gender_label = multi_class_label[3:5]
    #     age_label = multi_class_label[5:]
    #     return (
    #         torch.argmax(mask_label) * 6
    #         + torch.argmax(gender_label) * 3
    #         + torch.argmax(age_label) * 1
    #     )

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(
        self,
        data_dir,
        target_type,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, target_type, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(
        self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = transforms.Compose(
            [
                CenterCrop((400, 300)),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
