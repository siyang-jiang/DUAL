# Data Directory

This directory contains datasets and data splits for few-shot learning experiments.

## Structure

```
data/
├── datasets/           # Raw dataset files
│   ├── miniImageNet/   # miniImageNet dataset
│   ├── tieredImageNet/ # tieredImageNet dataset
│   ├── CIFAR-FS/       # CIFAR-FS dataset
│   └── CUB/            # CUB-200-2011 dataset
└── splits/             # Episode splits for few-shot learning
    ├── miniImageNet/   # Train/val/test splits
    ├── tieredImageNet/
    ├── CIFAR-FS/
    └── CUB/
```

## Supported Datasets

### miniImageNet
- **Description**: 100 classes with 600 examples per class
- **Split**: 64 training, 16 validation, 20 test classes
- **Image Size**: 84x84 RGB images
- **Download**: Follow instructions in `datasets/miniImageNet/README.md`

### tieredImageNet
- **Description**: 608 classes with variable examples per class
- **Split**: 351 training, 97 validation, 160 test classes  
- **Image Size**: 84x84 RGB images
- **Download**: Follow instructions in `datasets/tieredImageNet/README.md`

### CIFAR-FS
- **Description**: 100 classes with 600 examples per class
- **Split**: 64 training, 16 validation, 20 test classes
- **Image Size**: 32x32 RGB images
- **Download**: Follow instructions in `datasets/CIFAR-FS/README.md`

### CUB-200-2011
- **Description**: 200 bird species with ~30 images per class
- **Split**: 100 training, 50 validation, 50 test classes
- **Image Size**: Variable (resized to 84x84)
- **Download**: Follow instructions in `datasets/CUB/README.md`

## Data Preparation

1. Download the datasets following the instructions in each dataset directory
2. Run the preprocessing scripts to generate episode splits:
   ```bash
   python scripts/preprocess_data.py --dataset miniImageNet
   ```

## Episode Generation

Episodes are pre-generated for consistent evaluation across experiments. Each episode contains:
- N support examples per class (N-shot)
- M query examples per class
- K classes total (K-way)

Default configurations:
- 5-way 1-shot: 5 classes, 1 support + 15 query per class
- 5-way 5-shot: 5 classes, 5 support + 15 query per class