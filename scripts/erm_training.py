"""
Train a CNN using empirical risk minimization (non-episodic training).
"""

import torch
import torchvision

from configs import model_config, pretraining_config, experiment_config
from src.erm_trainer import ermTrainerResize, ermTrainer, ermTrainer_TFT
from src.erm_training_steps import get_data, get_model, train
from src.pretrain import get_unmixed_dataloader, pretrain_model
from src.running_steps import eval_model, prepare_output, set_and_print_random_seed
import wandb

import os


def main():
    prepare_output(test=False, ERM=True)
    set_and_print_random_seed()
    TRAINER = ermTrainerResize if experiment_config.MODEL != 3 else ermTrainer_TFT
    train_loader, val_loader, n_classes = get_data(two_stream=True)
    model = get_model(n_classes)
    trainer = TRAINER(model, train_loader, val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
