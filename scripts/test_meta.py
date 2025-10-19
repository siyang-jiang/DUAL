import os
from configs.experiment_config import (
    EXP_INFO,
    GPU_ID,
    GROUP,
    SMART_RESIZER,
    N_SHOT,
    MODEL,
    BATCH,
)
from src.running_steps import (
    eval_model,
    load_model,
    prepare_output,
    set_and_print_random_seed,
)
from src.erm_training_steps import get_model
from configs.list_config import QUEST_LIST, MODEL_LIST, LEVEL_LIST
from configs.dataset_config import IMAGE_SIZE, CLASSES
import sys
from pathlib import Path
from shutil import rmtree, move
import torch
from torch import nn
from loguru import logger

# from torchsummary import summary
import argparse
from configs import experiment_config, model_config, pretraining_config

stringmodule = __import__("string")
import wandb


def main(args=None, save_dir=None):
    torch.multiprocessing.set_sharing_strategy("file_system")
    experiment_group = GROUP
    experiment_name = EXP_INFO
    log_name = args["name"]
    logPath = Path(experiment_config.SAVE_DIR / f"{log_name}.log")
    if logPath.exists():
        with open(logPath) as read:
            lines = read.readlines()
        read.close()
        for l in lines:
            if "Accuracy" in l and not args["debug"]:
                logger.warning(
                    f"Test |{experiment_config.SAVE_DIR}| have already finished."
                )
                sys.exit(0)
    prepare_output(test=True, log_name=log_name)

    logger.warning(f"START TESTING")

    model_path = "outputs/baseline/meta_dataset/0target-0shot/TP/best_model.tar"
    model_R_path = ""
    val_model = load_model(
        model_path,
        episodic=True,
        use_fc=(MODEL == 3),
        force_ot=experiment_config.FORCE_OT,
    )  # Original, Disable OT here

    set_and_print_random_seed()

    model_R = get_model(1024).to(f"cuda:{GPU_ID}")  # TODO: Seperate R from model
    model_R.clf = nn.Identity()
    model_R.load_state_dict(torch.load(model_R_path), strict=False)
    repairer = model_R.R
    val_model.inverse_resize = repairer.to(f"cuda:{GPU_ID}")
    logger.warning("Repaier used")
    test_meta(val_model)


def test_meta(model):
    logger.info("loading meta dataset")
    from src.data_tools.datasets.meta_dataset.args import get_args_parser
    from src.data_tools.datasets.meta_dataset import config as config_lib
    from src.data_tools.datasets.meta_dataset.utils import Split
    from src.data_tools.datasets.meta_h5_dataset import FullMetaDatasetH5
    from src.data_tools.datasets.meta_val_dataset import MetaValDataset

    args_meta = get_args_parser().parse_args()
    args_meta.image_size = 128

    test_sources = [
        "traffic_sign",
        "mscoco",
        "ilsvrc_2012",
        "omniglot",
        "aircraft",
        "cu_birds",
        "dtd",
        "quickdraw",
        "fungi",
        "vgg_flower",
    ]
    acc_list, confidence_list = [], []
    for source in test_sources:
        logger.info(f"Testing on {source}")
        args_meta.test_sources = [source]
        testSet = FullMetaDatasetH5(args_meta, Split.TEST)
        logger.info(f"There are {len(testSet)} val_batch in total")
        # testSet = {source: testSet}
        test_loader = torch.utils.data.DataLoader(
            testSet,
            batch_size=1,
            num_workers=os.cpu_count(),
            pin_memory=False,
            drop_last=False,
        )
        acc, confidence = eval_model(model, test_loader)
        acc_list.append(acc)
        confidence_list.append(confidence)
    acc = sum(acc_list) / len(acc_list)
    confidence = sum(confidence_list) / len(confidence_list)
    logger.success(f"Final Average Test Accuracy = {acc:4.2f}% +- {confidence:4.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rewrite configs")
    parser.add_argument("--name", type=str, default="testing_R", help="logfile name")
    parser.add_argument("--debug", type=int, default=0, help="debug flag")
    parser.add_argument("--save_dir", type=str, default="", help="logfile name")
    args = vars(parser.parse_args())
    main(args, args["save_dir"])
