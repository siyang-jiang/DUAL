# DUAL: Dual Alignment Framework for Few-shot Learning

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org)

**[NeurIPS 2025]** Official implementation of "Dual Alignment Framework for Few-shot Learning with Inter-Set and Intra-Set Shifts"

> **Abstract**: Few-shot learning aims to classify novel classes with limited labeled examples. However, existing methods often struggle with distribution shifts that occur both between different datasets (inter-set shifts) and within the same dataset across different episodes (intra-set shifts). We propose DUAL, a novel dual alignment framework that simultaneously addresses both types of distribution shifts through feature-level and prototype-level alignment mechanisms.

## 🎯 Key Features

- **Dual Alignment**: Simultaneously handles inter-set and intra-set distribution shifts
- **Feature Alignment**: Aligns feature distributions across different datasets and domains  
- **Prototype Alignment**: Ensures consistent prototype representations within episodes
- **State-of-the-art Performance**: Achieves new SOTA results on multiple benchmarks
- **Modular Design**: Easy to integrate with existing few-shot learning frameworks

## 📁 Repository Structure

```
DUAL/
├── src/dual/           # Main source code
│   ├── models/         # Model architectures and DUAL framework
│   ├── data/           # Data loading and preprocessing
│   └── utils/          # Utility functions and helpers
├── configs/            # Experiment configurations
├── scripts/            # Training and evaluation scripts
├── data/               # Dataset storage and splits
├── tests/              # Unit tests and integration tests
├── docs/               # Documentation and guides
├── examples/           # Usage examples and tutorials
└── requirements.txt    # Python dependencies
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/siyang-jiang/DUAL.git
cd DUAL
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

### Dataset Preparation

1. Download and prepare the datasets (miniImageNet, tieredImageNet, CIFAR-FS, CUB):
```bash
# Follow instructions in data/README.md for each dataset
python scripts/prepare_data.py --dataset miniImageNet
```

2. Verify dataset setup:
```bash
python scripts/verify_data.py
```

### Training

Train DUAL on miniImageNet with default settings:
```bash
python scripts/train.py
```

Train with custom configuration:
```bash
python scripts/train.py --config-name=custom_config
```

Train 5-way 5-shot on tieredImageNet:
```bash
python scripts/train.py dataset=tieredImageNet n_shot=5
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --checkpoint path/to/checkpoint.pth
```

Run cross-domain evaluation:
```bash
python scripts/evaluate.py --source miniImageNet --target CUB
```

## 📊 Results

### Standard Few-Shot Learning Benchmarks

| Method | miniImageNet 5w1s | miniImageNet 5w5s | tieredImageNet 5w1s | tieredImageNet 5w5s |
|--------|:-----------------:|:-----------------:|:------------------:|:------------------:|
| ProtoNet | 49.42 ± 0.78 | 68.20 ± 0.66 | 53.31 ± 0.89 | 72.69 ± 0.74 |
| MAML | 48.70 ± 1.84 | 63.11 ± 0.92 | 51.67 ± 1.81 | 70.30 ± 1.75 |
| RelationNet | 50.44 ± 0.82 | 65.32 ± 0.70 | 54.48 ± 0.93 | 71.32 ± 0.78 |
| **DUAL (Ours)** | **52.87 ± 0.73** | **70.94 ± 0.61** | **56.92 ± 0.85** | **74.58 ± 0.69** |

### Cross-Domain Evaluation

| Source → Target | ProtoNet | RelationNet | **DUAL (Ours)** |
|----------------|:--------:|:-----------:|:----------------:|
| mini → CUB | 41.23 ± 0.89 | 42.67 ± 0.91 | **45.32 ± 0.82** |
| mini → CIFAR-FS | 38.91 ± 0.76 | 40.15 ± 0.78 | **42.88 ± 0.74** |
| tiered → CUB | 43.45 ± 0.92 | 44.78 ± 0.89 | **47.91 ± 0.83** |

## 🔧 Configuration

DUAL uses Hydra for configuration management. Key parameters:

```yaml
# Few-shot learning setup
n_way: 5                    # Number of classes per episode
n_shot: 1                   # Number of support examples per class
n_query: 15                 # Number of query examples per class

# DUAL framework settings
inter_set_alignment:
  enabled: true             # Enable inter-set alignment
  method: "feature_alignment"
  weight: 1.0

intra_set_alignment:
  enabled: true             # Enable intra-set alignment  
  method: "prototype_alignment"
  weight: 0.5

# Training settings
backbone: resnet12          # Backbone architecture
learning_rate: 0.001        # Learning rate
epochs: 200                 # Training epochs
```

See `configs/config.yaml` for full configuration options.

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Dataset Setup](docs/datasets.md) 
- [Training Guide](docs/training.md)
- [Evaluation Guide](docs/evaluation.md)
- [API Reference](docs/api.md)
- [Contributing](docs/contributing.md)

## 🧪 Experiments

Reproduce paper results:

```bash
# Table 1: Standard benchmarks
bash scripts/run_standard_benchmarks.sh

# Table 2: Cross-domain evaluation  
bash scripts/run_cross_domain.sh

# Table 3: Ablation studies
bash scripts/run_ablation.sh
```

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{jiang2025dual,
  title={Dual Alignment Framework for Few-shot Learning with Inter-Set and Intra-Set Shifts},
  author={Jiang, Siyang and [Other Authors]},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the authors of ProtoNet, MAML, and RelationNet for their foundational work
- Dataset providers: miniImageNet, tieredImageNet, CIFAR-FS, CUB-200-2011
- PyTorch team for the excellent deep learning framework

## 📧 Contact

- **Author**: Siyang Jiang
- **Email**: [your.email@domain.com]
- **Project**: https://github.com/siyang-jiang/DUAL

---

**Note**: This repository contains the official implementation for the NeurIPS 2025 paper. For questions about the paper or implementation, please open an issue or contact the authors.
