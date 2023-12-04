#!/usr/bin/env python3

"""Image transformations."""
import torchvision as tv
import random

# List of data augmentation techniques
augmentations = [
    tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    tv.transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    tv.transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    tv.transforms.RandomVerticalFlip(0.5),
    tv.transforms.RandomRotation(15),
    tv.transforms.RandomGrayscale(p=0.2),
    tv.transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    tv.transforms.RandomSolarize(threshold=128),
    tv.transforms.RandomPosterize(bits=4),
]

selected_augmentations = random.sample(augmentations, 2)

def get_transforms(split, size):
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if size == 448:
        resize_dim = 512
        crop_dim = 448
    elif size == 224:
        resize_dim = 256
        crop_dim = 224
    elif size == 384:
        resize_dim = 438
        crop_dim = 384
    if split == "train":
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                # *selected_augmentations,
                # tv.transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                # tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    return transform
