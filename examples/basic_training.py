# Basic Example: Training DUAL on miniImageNet

This example demonstrates how to train the DUAL framework on the miniImageNet dataset for 5-way 1-shot classification.

## Prerequisites

Make sure you have:
1. Installed the DUAL package: `pip install -e .`
2. Downloaded and prepared the miniImageNet dataset (see `data/README.md`)
3. CUDA-capable GPU (recommended)

## Code Example

```python
import torch
from omegaconf import OmegaConf
from dual.models import DualModel
from dual.data import MiniImageNetDataset, FewShotDataLoader
from dual.utils import set_seed, Logger

# Configuration
cfg = OmegaConf.load('configs/config.yaml')
set_seed(cfg.seed)

# Initialize model
model = DualModel(
    backbone=cfg.backbone,
    feature_dim=cfg.feature_dim,
    n_way=cfg.n_way,
    inter_alignment=cfg.inter_set_alignment,
    intra_alignment=cfg.intra_set_alignment
)

# Load dataset
dataset = MiniImageNetDataset(
    root='data/datasets/miniImageNet',
    split='train',
    transform=True
)

dataloader = FewShotDataLoader(
    dataset=dataset,
    n_way=cfg.n_way,
    n_shot=cfg.n_shot,
    n_query=cfg.n_query,
    n_episodes=cfg.n_episodes,
    batch_size=cfg.batch_size
)

# Setup training
device = torch.device(cfg.device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
logger = Logger(cfg.log_dir)

# Training loop
model.train()
for epoch in range(cfg.epochs):
    total_loss = 0.0
    total_acc = 0.0
    
    for episode, batch in enumerate(dataloader):
        # Move batch to device
        support_x = batch['support_x'].to(device)
        support_y = batch['support_y'].to(device) 
        query_x = batch['query_x'].to(device)
        query_y = batch['query_y'].to(device)
        
        # Forward pass
        logits = model(support_x, support_y, query_x)
        loss = torch.nn.functional.cross_entropy(logits, query_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = (logits.argmax(dim=1) == query_y).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
        
        if episode % cfg.log_interval == 0:
            print(f'Epoch {epoch}, Episode {episode}: Loss={loss.item():.4f}, Acc={acc.item():.4f}')
    
    # Log epoch results
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    logger.log({
        'epoch': epoch,
        'train_loss': avg_loss,
        'train_acc': avg_acc
    })
    
    print(f'Epoch {epoch}: Average Loss={avg_loss:.4f}, Average Acc={avg_acc:.4f}')

print("Training completed!")
```

## Running the Example

1. **Using the script directly**:
```bash
python examples/basic_training.py
```

2. **Using the main training script**:
```bash
python scripts/train.py
```

3. **With custom parameters**:
```bash
python scripts/train.py n_shot=5 epochs=100 learning_rate=0.0001
```

## Expected Output

```
Epoch 0, Episode 0: Loss=1.6094, Acc=0.2000
Epoch 0, Episode 100: Loss=1.4523, Acc=0.3200
Epoch 0, Episode 200: Loss=1.2156, Acc=0.4533
...
Epoch 0: Average Loss=1.3456, Average Acc=0.3789
...
Epoch 199: Average Loss=0.4567, Average Acc=0.8234
Training completed!
```

## Next Steps

- Try different datasets: `python scripts/train.py dataset=tieredImageNet`
- Experiment with 5-shot: `python scripts/train.py n_shot=5`
- Enable cross-domain evaluation: See `examples/cross_domain.py`
- Visualize results: See `examples/visualization.py`