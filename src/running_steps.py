from copy import deepcopy
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from collections import OrderedDict
from distutils.dir_util import copy_tree
from pathlib import Path
import random
from shutil import rmtree

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from configs import (
    dataset_config,
    evaluation_config,
    training_config,
    model_config,
    experiment_config,
)
from src.utils import set_device, elucidate_ids, get_episodic_loader
import wandb


def prepare_output(test=False, ERM=False, log_name="running"):
    save_dir = experiment_config.SAVE_DIR
    exp_name = experiment_config.EXP_INFO
    if not test:
        wandb.login()
        exp_name = experiment_config.EXP_INFO
        if ERM:
            save_dir = experiment_config.ERM_MODEL_DIR
            exp_name = f"{experiment_config.EXP_NAME} - ERM_training"
        wandb.init(
            entity="dual",
            project=experiment_config.BATCH,
            name=exp_name,
            tags=["training", dataset_config.DATASET.__name__],
        )
    logger.remove()
    logger.add(
        sys.stderr,
        format=f"<c>{experiment_config.GROUP} - {exp_name}</c>"
        + " |<g>{time: HH:mm:ss}</g> | - <lvl>{message}</lvl>",
    )
    if experiment_config.SAVE_RESULTS:
        if save_dir.exists() and test == False:
            logPath = Path(save_dir / "running.log")
            if logPath.exists():
                with open(logPath) as read:
                    lines = read.readlines()
                read.close()
                for l in lines:
                    if "2000 Test Accuracy" in l or "Training early stops" in l:
                        logger.warning("The experiment have already finished.")
                        sys.exit(0)
            # input("The experiment directory already exists. Press Enter to cover or Ctrl+C to exit.")
            logger.warning(
                "The experiment directory already exists but have not finished, delete all and restart."
            )
            rmtree(str(save_dir))
            logger.info(
                "Deleting previous content of {directory}",
                directory=save_dir,
            )
        save_dir.mkdir(parents=True, exist_ok=experiment_config.USE_POLYAXON)
        logger.add(
            save_dir / f"{log_name}.log",
            format=f"<c>{experiment_config.GROUP} - {exp_name}</c>"
            + " |{time:YYYY-MM-DD HH:mm:ss}|{elapsed}| <lvl>{module}</lvl> - {message}",
        )
        if not test:
            copy_tree("configs", str(save_dir / "experiment_parameters"))
        dataset_name = (
            dataset_config.DATASET.__name__
            if hasattr(dataset_config.DATASET, "__name__")
            else dataset_config.DATASET.func.__name__
        )
        logger.info(f"Experiment Information: {experiment_config.GROUP} - {exp_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.success(f"Experiment Describe: {experiment_config.DESCRIBE}")
        logger.info(
            "Parameters and outputs of this experiment will be saved in {directory}",
            directory=save_dir,
        )
    else:
        logger.info("This experiment will not be saved on disk.")


def set_and_print_random_seed():
    """
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Returns:
        int: numpy random seed

    """
    random_seed = experiment_config.RANDOM_SEED
    if not random_seed:
        random_seed = np.random.randint(0, 2**32 - 1)
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        random.seed(np.random.randint(0, 2**32 - 1))
    else:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed : {random_seed}")

    return random_seed


def train_model():
    logger.info("Model training started.")
    logger.info(
        f"{training_config.N_TARGET} target, {training_config.N_WAY} way - {training_config.N_SOURCE} shot"
    )
    logger.info(f"Backbone: {model_config.BACKBONE}")
    logger.info(f"Model: {model_config.MODEL}")
    logger.info(f"Seed: {experiment_config.RANDOM_SEED}")
    logger.info("Initializing data loaders...")
    train_loader, _ = get_episodic_loader(
        "train",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_EPISODES,
    )
    val_loader, _ = get_episodic_loader(
        "val",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_VAL_TASKS,
    )
    if training_config.TEST_SET_VALIDATION_FREQUENCY:
        test_loader, _ = get_episodic_loader(
            "test",
            n_way=training_config.N_WAY,
            n_source=training_config.N_SOURCE,
            n_target=training_config.N_TARGET,
            n_episodes=training_config.N_VAL_TASKS,
        )

    logger.info("Initializing model...")

    model = set_device(model_config.MODEL(model_config.BACKBONE))
    optimizer = training_config.OPTIMIZER(model.parameters())

    max_acc = -1.0
    best_model_epoch = -1
    best_model_state = None
    save_dir = experiment_config.SAVE_DIR
    writer = SummaryWriter(log_dir=save_dir)

    logger.info("Model and data are ready. Starting training...")
    for epoch in range(training_config.N_EPOCHS):  # SY: seeing the batch size
        # Set model to training mode
        model.train()
        # Execute a training loop of the model
        train_loss, train_acc = model.train_loop(epoch, train_loader, optimizer)
        writer.add_scalar("Train/loss", train_loss, epoch)
        writer.add_scalar("Train/acc", train_acc, epoch)
        # Set model to evaluation mode
        model.eval()
        # Evaluate on validation set
        torch.cuda.empty_cache()
        val_loss, val_acc, _ = model.eval_loop(val_loader)
        writer.add_scalar("Val/loss", val_loss, epoch)
        writer.add_scalar("Val/acc", val_acc, epoch)

        # We make sure the best model is saved on disk, in case the training breaks
        if val_acc > max_acc:
            max_acc = val_acc
            best_model_epoch = epoch
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_dir / "best_model.tar")

        if training_config.TEST_SET_VALIDATION_FREQUENCY:
            if (
                epoch % training_config.TEST_SET_VALIDATION_FREQUENCY
                == training_config.TEST_SET_VALIDATION_FREQUENCY - 1
            ):
                logger.info("Validating on test set...")
                _, test_acc, _ = model.eval_loop(test_loader)
                writer.add_scalar("Test/acc", test_acc, epoch)

    logger.info(f"Training over after {training_config.N_EPOCHS} epochs")
    logger.info("Retrieving model with best validation accuracy...")
    model.load_state_dict(best_model_state)
    logger.info(f"Retrieved model from epoch {best_model_epoch}")

    writer.close()

    return model


def train_model_meta():
    logger.info("Model training started.")
    logger.info(
        f"{training_config.N_TARGET} target, {training_config.N_WAY} way - {training_config.N_SOURCE} shot"
    )
    logger.info(f"Backbone: {model_config.BACKBONE}")
    logger.info(f"Model: {model_config.MODEL}")
    logger.info(f"Seed: {experiment_config.RANDOM_SEED}")
    logger.info("Initializing data loaders...")
    from src.data_tools.datasets.meta_dataset.args import get_args_parser
    from src.data_tools.datasets.meta_dataset.utils import Split
    from src.data_tools.datasets.meta_h5_dataset import FullMetaDatasetH5
    from src.data_tools.datasets.meta_val_dataset import MetaValDataset

    args = get_args_parser().parse_args()
    args.base_sources = ["ilsvrc_2012"]

    trainSet = FullMetaDatasetH5(args, Split.TRAIN)
    args.nValEpisode = 120
    # valSet = MetaValDataset(args, Split.VALID)
    valSet = FullMetaDatasetH5(args, Split.VALID)
    # valSet = MetaValDataset(
    #     os.path.join(args.data_path, "ilsvrc_2012", f"val_ep{args.nValEpisode}_img{args.image_size}.h5"), num_episodes=args.nValEpisode
    # )
    train_dataloader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=1,
        num_workers=os.cpu_count(),
        pin_memory=False,
        drop_last=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        valSet,
        batch_size=1,
        num_workers=os.cpu_count(),
        pin_memory=False,
        drop_last=False,
    )
    logger.info("Initializing model...")
    model = set_device(model_config.MODEL(model_config.BACKBONE))
    optimizer = training_config.OPTIMIZER(model.parameters())
    max_acc = -1.0
    best_model_epoch = -1
    best_model_state = None
    save_dir = experiment_config.SAVE_DIR
    logger.info("Model and data are ready. Starting training...")
    for epoch in range(training_config.N_EPOCHS):  # SY: seeing the batch size
        # Set model to training mode
        model.train()
        # Execute a training loop of the model
        train_loss, train_acc = model.train_loop(epoch, train_dataloader, optimizer)

        # Set model to evaluation mode
        model.eval()
        torch.cuda.empty_cache()
        # Evaluate on validation set
        val_loss, val_acc, _ = model.eval_loop(val_dataloader)
        val_acc = val_acc[0] if isinstance(val_acc, tuple) else val_acc

        # We make sure the best model is saved on disk, in case the training breaks
        if val_acc > max_acc:
            max_acc = val_acc
            best_model_epoch = epoch
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_dir / "best_model.tar")

    logger.info(f"Training over after {training_config.N_EPOCHS} epochs")
    logger.info("Retrieving model with best validation accuracy...")
    model.load_state_dict(best_model_state)
    logger.info(f"Retrieved model from epoch {best_model_epoch}")
    return model


def load_model_episodic(
    model: nn.Module, state_dict: OrderedDict
) -> nn.Module:  # episodic
    model.load_state_dict(state_dict, strict=False)
    return model


def load_model_non_episodic(
    model: nn.Module, state_dict: OrderedDict, use_fc: bool
) -> nn.Module:
    if use_fc:
        # model.feature.clf = set_device(
        #     nn.Linear(
        #         model.feature.final_feat_dim,
        #         dataset_config.CLASSES["train"] + dataset_config.CLASSES["val"],
        #     )
        # )
        model.feature.trunk.fc = set_device(
            nn.Linear(
                model.feature.final_feat_dim,
                dataset_config.CLASSES["train"] + dataset_config.CLASSES["val"],
            )
        )

    state_keys = list(state_dict.keys())
    for _, key in enumerate(state_keys):
        if "clf." in key:
            newkey = key.replace("clf.", "trunk.fc.")
            state_dict[newkey] = state_dict.pop(key)

    model.feature.load_state_dict(
        OrderedDict(
            [
                (k, v)
                for k, v in state_dict.items()
                if "H." not in k and "clf_SIMCLR." not in k and "R.core" not in k
            ]
        )
        if use_fc
        else OrderedDict(
            [
                (k, v)
                for k, v in state_dict.items()
                if "fc" not in k
                and "H." not in k
                and "clf_SIMCLR." not in k
                and "R.core" not in k
            ]
        )
    )

    # model.feature.load_state_dict(
    #     state_dict
    #     if use_fc
    #     else OrderedDict([(k, v) for k, v in state_dict.items() if ".fc." not in k])
    # )
    return model


def load_model(
    state_path: Path, episodic: bool, use_fc: bool, force_ot: bool
) -> nn.Module:
    model = set_device(model_config.MODEL(model_config.BACKBONE))

    if force_ot:
        model.transportation_module = model_config.TRANSPORTATION_MODULE
        logger.info("Forced the Optimal Transport module into the model.")
    else:
        logger.info("No Optimal Transport module.")

    state_dict = torch.load(state_path)
    model = (
        load_model_episodic(model, state_dict)
        if episodic
        else load_model_non_episodic(model, state_dict, use_fc)
    )

    logger.info(f"Loaded model from {state_path}")

    return model


def eval_model(model, test_loader=None):
    logger.info("Initializing test data...")
    if test_loader is None:
        test_loader, test_dataset = get_episodic_loader(
            "test",
            n_way=evaluation_config.N_WAY_EVAL,
            n_source=evaluation_config.N_SOURCE_EVAL,
            n_target=evaluation_config.N_TARGET_EVAL,
            n_episodes=evaluation_config.N_TASKS_EVAL,
        )
    # logger.info(f"Testing resize level:{experiment_config.TESTING_QUERY_RESIZE_FACTOR}")
    # logger.info(f"Testing resize module:{experiment_config.TESTING_QUERY_TRANSFROM}")
    logger.info(f"GPU_ID:{experiment_config.GPU_ID}")
    logger.info(f"REPAIER:{experiment_config.SMART_RESIZER}")
    logger.info("Starting model evaluation...")
    model.eval()

    _, acc, stats_df = model.eval_loop(test_loader)

    # stats_df = elucidate_ids(stats_df, test_dataset)
    # stats_df.to_csv(experiment_config.SAVE_DIR / "evaluation_stats.csv", index=False)
    # writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)
    # writer.add_scalar("Evaluation accuracy", acc)
    # writer.close()
    return acc
