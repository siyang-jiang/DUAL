from functools import partial
import torch

# Global config for the experiment
from torchvision import transforms
from configs.dataset_config import IMAGE_SIZE
from configs.dataset_specs.cifar_100_c.perturbation_params import (
    PERTURBATION_PARAMS,
)  # TODO: Decouple this
import src.data_tools.perturbations as perturbations
import random
import math

LEVEL_DICT = {
    "resize": [2, 4, 8],
    "blur": [3, 5, 7],
    "noise": [1e-4, 5e-4, 1e-3],
    "colorjitter": [0.5, 0.75, 1],
}


def gaussian_transfrom(input, params, gpu):
    gaussian_partial = partial(
        getattr(perturbations, "gaussian_noise"),
        severity_params=params,
        image_size=IMAGE_SIZE,
    )
    return torch.tensor(gaussian_partial(input.cpu())).float().to(f"cuda:{gpu}")


def get_randomchoice(gpu, level):
    quest_list = [
        [
            transforms.Resize(IMAGE_SIZE // LEVEL_DICT["resize"][i]),
            transforms.GaussianBlur(kernel_size=LEVEL_DICT["blur"][i]),
            partial(gaussian_transfrom, params=LEVEL_DICT["noise"][i], gpu=gpu),
            transforms.ColorJitter(
                brightness=LEVEL_DICT["colorjitter"][i],
                contrast=LEVEL_DICT["colorjitter"][i],
                saturation=LEVEL_DICT["colorjitter"][i],
            ),
        ]
        for i in range(3)
    ]
    if level == 0:
        return transforms.RandomChoice(quest_list[0])
    if level == 1:
        return transforms.RandomChoice(quest_list[1])
    if level == 2:
        return transforms.RandomChoice(quest_list[2])


def apply(transform_list, p=0.5):
    output = []
    # for t in transform_list:
    #     output.append(transforms.RandomApply(torch.nn.ModuleList([t]),p))
    return output


def get_randomapply(gpu, level):
    quest_list = [
        [
            transforms.Resize(IMAGE_SIZE // LEVEL_DICT["resize"][i]),
            transforms.GaussianBlur(kernel_size=LEVEL_DICT["blur"][i]),
            partial(gaussian_transfrom, params=LEVEL_DICT["noise"][i], gpu=gpu),
            transforms.ColorJitter(
                brightness=LEVEL_DICT["colorjitter"][i],
                contrast=LEVEL_DICT["colorjitter"][i],
                saturation=LEVEL_DICT["colorjitter"][i],
            ),
        ]
        for i in range(3)
    ]
    if level == 0:
        return torch.nn.Sequential(*apply(quest_list[0]))
    if level == 1:
        return torch.nn.Sequential(*apply(quest_list[1]))
    if level == 2:
        return torch.nn.Sequential(
            *apply(quest_list[0] + quest_list[1] + quest_list[2])
        )


def get_randomchoice_MKII(gpu, level):  # Deprecated
    return transforms.ToTensor
    quest_list = [
        get_partial(perturbation_name, level, gpu)
        for perturbation_name in perturbation_name_list
    ]
    # quest_list += [transforms.Resize(IMAGE_SIZE//LEVEL_DICT["resize"][level])]
    return transforms.RandomChoice(quest_list)


def get_randomapply_MKII(gpu, level):  # Deprecated
    return transforms.ToTensor
    quest_list = [
        get_partial(perturbation_name, level, gpu)
        for perturbation_name in perturbation_name_list
    ]
    return torch.nn.Sequential(*apply(quest_list))


def get_transform(quest, level, gpu):
    transform_list = [
        transforms.Resize(IMAGE_SIZE // LEVEL_DICT["resize"][level]),
        transforms.GaussianBlur(kernel_size=LEVEL_DICT["blur"][level]),
        partial(gaussian_transfrom, params=LEVEL_DICT["noise"][level], gpu=gpu),
        transforms.ColorJitter(
            brightness=LEVEL_DICT["colorjitter"][level],
            contrast=LEVEL_DICT["colorjitter"][level],
            saturation=LEVEL_DICT["colorjitter"][level],
        ),
        get_randomchoice_MKII(gpu, level),
        get_randomapply_MKII(gpu, level),
        transforms.ToTensor,
        transforms.ToTensor,
    ]
    return transform_list[quest]


## 2023
## DONE: Switch to the dataloader implementation.
""" Deprecated
perturbation_name_list = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "gaussian_blur",
    "glass_blur",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "fog",
    "frost",
    "snow",
    "spatter",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "saturate",
]


def partial_name_to_parameter_name(input_string):
    if input_string == "elastic_transform":
        return "Elastic"
    elif input_string == "jpeg_compression":
        return "JPEG"
    else:
        return input_string.replace("_", " ").title()


def get_partial(perturbation_name, level, gpu):
    parameter_list = PERTURBATION_PARAMS[partial_name_to_parameter_name(perturbation_name)]
    level = math.ceil((level + 1) / 3 * len(parameter_list))  # LEVEL index starts from 0
    temp_partial = partial(
        getattr(perturbations, perturbation_name),
        severity_params=parameter_list[level],
        image_size=IMAGE_SIZE,
    )
    return partial(tensorize, pertubrbation=temp_partial, gpu=gpu)


totensor = transforms.ToTensor()
topilimage = transforms.ToPILImage()


def tensorize(input, pertubrbation, gpu):
    return totensor(pertubrbation(topilimage(input))).float().to(f"cuda:{gpu}")
"""
