# DUAL: Dual Alignment Framework for Few-shot Learning with Inter-Set and Intra-Set Shifts

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://neurips.cc/virtual/2025/poster/115878)
[![Scholar Award](https://img.shields.io/badge/Award-Scholar%20Award-gold.svg)](https://neurips.cc/Conferences/2025/FinancialAssistance)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/siyang-jiang/DUAL)
[![Project Page](https://img.shields.io/badge/Project-Page-orange)](https://siyang-jiang.github.io/DUAL)

> **DUAL** addresses the critical challenge of **Dual Support-Query Shift (DSQS)** in few-shot learning, where both support and query sets experience distribution shifts. Our framework employs adversarial training and dual optimal transport to achieve robust alignment across both inter-set and intra-set shifts.

## 🎯 Key Contributions

- **Novel Problem Formulation**: First to tackle **Dual Support-Query Shift (DSQS)** where both support and query sets are shifted
- **Dual Alignment Mechanism**: Two-stage optimal transport for intra-set and inter-set alignment
- **Adversarial Robustness**: Generator-based adversarial training for robust feature learning
- **Pixel-Level Repair**: Smart resizer network to mitigate corruption effects
- **State-of-the-Art Results**: Superior performance across multiple benchmarks under realistic distribution shifts

## 📰 News

- 🔥 **[2025-10]** Code and project page released
- 🏆 **[2025-10]** **Scholar Award** at **NeurIPS 2025**
- 🔥 **[2025-09]** DUAL accepted by **NeurIPS 2025**


## 🚀 Quick Start

### Environment Setup
**Recommended**: Python 3.9+
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration
Launch `scripts/run_exp.py` to automatically copy `configs_template/` to `configs/` with proper setup.

### Full Pipeline Example
**mini-ImageNet with ProtoNet**:
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

**Quick Demo**: Run `./run.sh` for the default mini-ImageNet workflow (0-shot ERM + 1/5-shot evaluation).

## 🔧 Configuration & Usage

### Dataset Configuration
Edit dataset roots and preprocessing options in `configs_template/all_datasets_configs/`. Perturbation recipes are located under `configs_template/dataset_specs/<dataset>/jsons`.

### Key Configuration Switches
- **`PGADA`** (flag `--noPGADA` disables the PGADA trainer) — `configs/experiment_config.py`
- **`SMART_RESIZER`** (controls the test-time repairer) — `configs/experiment_config.py`
- **`PROPORTION`** and **`MULTI_PERTUBATION`** (perturbation strength and sampling) — `configs/experiment_config.py`
- **`SIMCLR`** (self-supervised branch during ERM training; override with `--SIMCLR 0`) — `configs/erm_training_config.py`
- **`FORCE_OT`** (force-enable or disable optimal transport during testing) — `configs/experiment_config.py`
- **`BACKBONE`** and **`R`** (feature extractor and repairer definitions) — `configs/model_config.py`

### Evaluation Modes
- **`--testing_lrs`**: Evaluation with the repairer enabled
- **`--testing`**: Evaluation without SMART_RESIZER
- **`--testing_ot 0`**: Evaluation without optimal transport
- **Logging**: Weights & Biases support enabled by default (configure in `configs/list_config.py`)

## 🏗️ Repository Structure

```text
DUAL/
├── configs_template/    # Default configuration templates copied to configs/
├── scripts/             # Training, evaluation, and experiment orchestration
├── src/                 # Core implementation: methods, modules, trainers, data tools
│   ├── methods/         # Few-shot learning algorithms
│   ├── modules/         # Neural network components  
│   ├── trainers/        # Training loops and optimization
│   └── data_tools/      # Data processing utilities
├── docs/                # Paper and documentation
├── run.sh               # End-to-end example script for mini-ImageNet
├── requirements.txt     # Python dependencies
└── index.html           # Project page
```

## 🔬 Method Overview

### Problem: Dual Support-Query Shift (DSQS)
Unlike traditional few-shot learning that assumes clean support sets, **DSQS** considers realistic scenarios where both support and query sets experience distribution shifts due to:
- Environmental variations (lighting, weather)
- Sensor noise and artifacts
- Domain adaptation challenges
- Data corruption and perturbations

### Solution: Dual Alignment Framework
1. **Pixel-Level Repair**: Smart resizer network `R(·)` mitigates corruption effects
2. **Adversarial Training**: Generator `G(·)` creates challenging examples for robust feature learning
3. **Intra-Set Alignment**: Optimal transport aligns samples within each set to class centroids
4. **Inter-Set Alignment**: Second optimal transport bridges the gap between support and query distributions

## 📊 Experimental Results

### Performance Under Distribution Shifts

| Dataset | Method | 1-shot | 5-shot |
|---------|--------|--------|--------|
| **CIFAR-100** | ProtoNet | 42.3±0.8 | 58.1±0.9 |
| | PGADA | 45.7±0.9 | 62.4±0.8 |
| | **DUAL (Ours)** | **48.9±0.8** | **65.7±0.7** |
| **mini-ImageNet** | ProtoNet | 49.4±0.8 | 68.2±0.7 |
| | PGADA | 52.3±0.9 | 71.5±0.6 |
| | **DUAL (Ours)** | **55.8±0.7** | **74.2±0.6** |

### Key Insights
- **Consistent Improvements**: DUAL outperforms state-of-the-art methods across all datasets and shot settings
- **Robustness**: Maintains superior performance even under severe distribution shifts
- **Efficiency**: Reasonable computational overhead with significant accuracy gains

## ⚠️ Troubleshooting

- **File Descriptor Issues**: Ensure `ulimit -n` is sufficiently high; `run.sh` sets it to one million to avoid dataloader issues
- **Configuration Problems**: Verify that `scripts/run_exp.py` completed the copy-and-rewrite step instead of terminating early
- **Memory Issues**: `--testing_lrs` loads an extra repairer model; plan GPU memory accordingly
- **Dependencies**: Check Python version (3.9+ recommended) and ensure all requirements are installed

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Share experimental results

## 📄 License

Released under the MIT License (see `LICENSE`).

## 🙏 Acknowledgments

This work was supported by research grants from National Taiwan University, CUHK, and SUNY Buffalo. Special thanks to NeurIPS 2025 for the Scholar Award recognition.

## 📖 Citation

If you use DUAL in your research, please cite our paper:

```bibtex
@inproceedings{jiang2025dual,
  title     = {Dual Alignment Framework for Few-shot Learning with Inter-Set and Intra-Set Shifts},
  author    = {Jiang, Siyang and Fang, Rui and Chen, Hsi-Wen and Ding, Wei and Xing, Guoliang and Chen, Ming-Syan},
  booktitle = {Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year      = {2025}
}
```

### Related Publications
- **PGADA**: [Perturbation-Guided Adversarial Alignment for Few-Shot Learning](https://arxiv.org/abs/2205.03817) (PAKDD 2022 Best Paper)
- **ArtFL**: [Exploiting Data Resolution in Federated Learning](https://github.com/siyang-jiang/ArtFL) (IPSN 2024)
- **CUHK-X**: [Large-Scale Multimodal Dataset for Human Action Recognition](https://github.com/siyang-jiang/CUHK-X) (SenSys 2025)

---

For questions or collaborations, please contact [siyangjiang@cuhk.edu.hk](mailto:siyangjiang@cuhk.edu.hk)
