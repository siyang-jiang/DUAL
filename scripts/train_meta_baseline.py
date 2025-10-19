"""
Run a complete experiment (training + evaluation)
"""

import os
from loguru import logger
import torch
import warnings
from test_meta import test_meta

warnings.filterwarnings("ignore")

from src.running_steps import (
    train_model,
    train_model_meta,
    eval_model,
    set_and_print_random_seed,
    prepare_output,
)

if __name__ == "__main__":
    prepare_output()
    torch.multiprocessing.set_sharing_strategy("file_system")
    set_and_print_random_seed()
    trained_model = train_model_meta()
    torch.cuda.empty_cache()

    set_and_print_random_seed()

    ### Test on meta-dataset
    from src.data_tools.datasets.meta_dataset.args import get_args_parser
    from src.data_tools.datasets.meta_dataset.utils import Split
    from src.data_tools.datasets.meta_h5_dataset import FullMetaDatasetH5

    test_meta(trained_model)
    # acc, test_loader = eval_model(val_model)
    # wandb.log({"test_acc": acc})
