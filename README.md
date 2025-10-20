# Dual Alignment (DUAL) Few-Shot Learning Framework

This repository contains the official implementation of the paper *Dual Alignment Framework for Few-shot Learning with Inter-Set and Intra-Set Shifts*. DUAL aligns both inter-set and intra-set distribution shifts, delivering a robust baseline for few-shot learning under realistic corruptions. The code covers data preparation, ERM pre-training, meta adaptation, and a range of ablations.

### News

- 🔥 [2025-10] Release Code of DUAL
- 🏆 [2025-10] **Scholar Award** at **NeurIPS 2025**
- 🔥 [2025-09] DUAL accepted by **NeurIPS 2025**


## Quick Start
- **Environment**: Recommended Python 3.9.
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- **Configuration templates**: Launch the `scripts/run_exp.py` scripts will copy `configs_template/` to `configs/`. 
- **Full pipeline example** (mini-ImageNet with ProtoNet):
  ```bash
  # ERM pre-training (0-shot target)
  python3 scripts/run_exp.py --dataset 0 --model 0 --shot 0 --target 0 --batch MAIN --gpu 0

  # Few-shot evaluation with SMART_RESIZER repairer
  for shot in 1 5; do
    for target in 8 16; do
      python3 scripts/run_exp.py \
        --dataset 0 --model 0 --shot ${shot} --target ${target} \
        --batch MAIN --gpu 0 --testing_lrs
    done
  done
  ```
- You can run `./run.sh` for the default mini-ImageNet workflow (0-shot ERM + 1/5-shot evaluation).

## Datasets and Configuration
- Edit dataset roots and preprocessing options in `configs_template/all_datasets_configs/`. Perturbation recipes live under `configs_template/dataset_specs/<dataset>/jsons`.
    - TODO: upload pre-packaged dataset.
- Frequently adjusted switches:
  - `PGADA` (flag `--noPGADA` disables the PGADA trainer) — `configs/experiment_config.py`
  - `SMART_RESIZER` (controls the test-time repairer) — `configs/experiment_config.py`
  - `PROPORTION` and `MULTI_PERTUBATION` (perturbation strength and sampling) — `configs/experiment_config.py`
  - `SIMCLR` (self-supervised branch during ERM training; override with `--SIMCLR 0`) — `configs/erm_training_config.py`
  - `FORCE_OT` (force-enable or disable optimal transport during testing) — `configs/experiment_config.py`
  - `BACKBONE` and `R` (feature extractor and repairer definitions) — `configs/model_config.py`

## Workflow
- **ERM pre-training**: `scripts/run_exp.py` launches ERM training with DuaL (`scripts.erm_training`).
- **Few-shot evaluation**:
  - `--testing_lrs`: evaluation with the repairer enabled.
  - `--testing`: evaluation without SMART_RESIZER.
  - `--testing_ot 0`: evaluation without optimal transport.
- **Logging**: Weights & Biases support is enabled by default. Configure project and group names in `configs/list_config.py` or disable if needed.

## Repository Layout
```text
.
├── configs_template/    # Default configuration templates copied to configs/
├── scripts/             # Training, evaluation, and experiment orchestration
├── src/                 # Core implementation: methods, modules, trainers, data tools
├── run.sh               # End-to-end example script for mini-ImageNet
├── requirements.txt     # Python dependencies
└── wandb/               # Default wandb logging directory (optional)
```

## Troubleshooting
- Ensure `ulimit -n` is sufficiently high; `run.sh` sets it to one million to avoid dataloader issues.
- If configurations seem outdated, verify that `scripts/run_exp.py` completed the copy-and-rewrite step instead of terminating early.
- `--testing_lrs` loads an extra repairer model; plan GPU memory accordingly.

## Citation
Please cite the paper if you use this codebase in your research:
```bibtex
@inproceedings{jiang2025dual,
  title     = {Dual Alignment Framework for Few-shot Learning with Inter-Set and Intra-Set Shifts},
  author = {Jiang, Siyang and Fang, Rui and Chen, Hsi-Wen and Ding, Wei and Xing, Guoliang and Chen, Ming-syan},
  booktitle = {Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year      = {2025}
}
```

## License
- Released under the MIT License (see `LICENSE`).
