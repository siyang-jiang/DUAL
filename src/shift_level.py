# Resize GaussianBlur Noise Colorjitter


import sys
import torch
import torchvision
from loguru import logger

# from torchsummary import summary
import argparse
from configs import experiment_config, model_config, pretraining_config
import importlib

stringmodule = __import__("string")
from configs.dataset_config import IMAGE_SIZE, CLASSES
from configs.experiment_config import EXP_INFO, GPU_ID, GROUP, SMART_RESIZER
from src.erm_training_steps import get_data, get_model
from src.running_steps import (
    eval_model,
    load_model,
    prepare_output,
    set_and_print_random_seed,
)


def main(args=None):
    experiment_group = GROUP
    experiment_name = EXP_INFO
    prepare_output(test=True, log_name=args["name"])
    logger.warning(f"START TESTING")
    set_and_print_random_seed()
    n_classes = torch.tensor(list(CLASSES.values())[:-1]).sum()
    model_path = f"outputs/{experiment_group}/{experiment_name}/model.pt"

    model_R = get_model(n_classes).to(f"cuda:{GPU_ID}")  # TODO: Seperate R from model
    model_R.load_state_dict(torch.load(model_path))

    val_model = load_model(
        model_path, episodic=False, use_fc=False, force_ot=True
    )  # Original
    # val_model.feature.trunk.fc = model_R.clf
    # val_model = model_config.MODEL(model_config.BACKBONE).to(f"cuda:{GPU_ID}")
    # val_model.feature = model_R

    set_and_print_random_seed()

    if SMART_RESIZER:
        val_model.inverse_resize = model_R.R.to(f"cuda:{GPU_ID}")
    acc, test_loader = eval_model(val_model)
    print(acc)
    # trainer.print_test_images(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rewrite configs")
    # parser.add_argument('--resize_factor', type = int, default=2, help='Change the resize factor')
    parser.add_argument("--name", type=str, default="testing", help="logfile name")
    args = vars(parser.parse_args())
    main(args)
