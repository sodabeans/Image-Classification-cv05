# python train.py --model ResNet --criterion focal;

# python train.py --model ResNet --criterion focal --dataset MaskSplitByProfileDataset;

# python train.py --model ResNet --criterion f1;

# python train.py --model ResNet --criterion f1 --dataset MaskSplitByProfileDataset;

# python train.py --model ResNet --criterion cross_entropy;

# python train.py --model ResNet --criterion cross_entropy --dataset MaskSplitByProfileDataset;

# python train.py --model ResNet --criterion label_smoothing;

# python train.py --model ResNet --criterion label_smoothing --dataset MaskSplitByProfileDataset;


# python train.py --model AlexNet --criterion focal;

# python train.py --model AlexNet --criterion focal --dataset MaskSplitByProfileDataset;

# python train.py --model AlexNet --criterion f1;

# python train.py --model AlexNet --criterion f1 --dataset MaskSplitByProfileDataset;

# python train.py --model AlexNet --criterion cross_entropy;

# python train.py --model AlexNet --criterion cross_entropy --dataset MaskSplitByProfileDataset;

# python train.py --model AlexNet --criterion label_smoothing;

# python train.py --model AlexNet --criterion label_smoothing --dataset MaskSplitByProfileDataset;


python train.py --model MultiTaskModel --dataset MultiTaskDataset --criterion focal;

python train.py --model MultiTaskModel --dataset MultiTaskDataset --criterion cross_entropy;

python train.py --model MultiTaskModel --dataset MultiTaskDataset --criterion f1;

python train.py --model MultiTaskModel --dataset MultiTaskDataset --criterion label_smoothing;
