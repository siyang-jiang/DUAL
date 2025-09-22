#!/usr/bin/env python3
"""
Training script for DUAL few-shot learning framework.
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path to import dual module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import dual


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    print("DUAL Training Script")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # TODO: Implement training logic
    print("Training logic will be implemented here...")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"N-way: {cfg.n_way}, N-shot: {cfg.n_shot}")
    print(f"Device: {cfg.device}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()