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
from distutils.dir_util import copy_tree
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
    n_classes = torch.tensor(list(CLASSES.values())[:-1]).sum()
    model_path = Path(experiment_config.ERM_MODEL_DIR) / "model.pt"
    val_model = load_model(
        model_path,
        episodic=False,
        use_fc=(MODEL == 3),
        force_ot=experiment_config.FORCE_OT,
    )  # Original, Disable OT here

    set_and_print_random_seed()

    if SMART_RESIZER:
        model_R = get_model(n_classes).to(f"cuda:{GPU_ID}")
        model_R.load_state_dict(torch.load(model_path))
        repairer = model_R.R
        val_model.inverse_resize = repairer.to(f"cuda:{GPU_ID}")
        logger.warning("Repaier used")
    else:
        logger.warning("Without Repaier")
    acc = eval_model(val_model)
    # wandb.log({"test_acc": acc})
    print(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rewrite configs")
    # parser.add_argument('--resize_factor', type = int, default=2, help='Change the resize factor')
    parser.add_argument(
        "--name", type=str, default="testing_trival", help="logfile name"
    )
    parser.add_argument("--debug", type=int, default=0, help="debug flag")
    parser.add_argument("--save_dir", type=str, default="", help="logfile name")
    args = vars(parser.parse_args())
    main(args, args["save_dir"])
