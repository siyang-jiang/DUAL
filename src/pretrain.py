import torch
from loguru import logger
from tqdm import tqdm
from src.utils import set_device
from configs import dataset_config, erm_training_config, pretraining_config
from src.utils import set_device, get_episodic_loader
from torch.utils.data import DataLoader
from .erm_training_steps import get_few_shot_split


def get_unmixed_dataloader(two_stream=False) -> tuple([DataLoader, DataLoader, int]):
    logger.info("Initializing data loaders...")
    train_set, val_set = get_few_shot_split(two_stream)
    if dataset_config.DATASET.__name__ == "FEMNIST":  # TODO : support FEMNIST
        pass
        # train_loader, train_set = get_episodic_loader(
        #     "train",
        #     n_way=32,
        #     n_source=1,
        #     n_target=1,
        #     n_episodes=200,
        # )
        # val_loader, val_set = get_episodic_loader(
        #     "val",
        #     n_way=training_config.N_WAY,
        #     n_source=training_config.N_SOURCE,
        #     n_target=training_config.N_TARGET,
        #     n_episodes=training_config.N_VAL_TASKS,
        # )
        # # Assume that train and val classes are entirely disjoints
        # n_classes = len(train_set.id_to_class)

    else:
        train_loader = DataLoader(
            train_set,
            batch_size=erm_training_config.BATCH_SIZE,
            num_workers=erm_training_config.N_WORKERS,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=erm_training_config.BATCH_SIZE,
            num_workers=erm_training_config.N_WORKERS,
        )
    return train_loader, val_loader


def pretrain_model(model, train_dataloader, val_dataloader):
    training_dataset_size = len(train_dataloader.dataset)
    val_dataset_size = len(val_dataloader.dataset)
    model = set_device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=pretraining_config.LR)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=2
    )

    logger.info("Pretraining model...")

    CEL = torch.nn.CrossEntropyLoss()
    # pbar = tqdm(range(batch_per_epoch))
    # pbar_val = tqdm(val_dataloader)
    for epoch in range(pretraining_config.MAX_EPOCH):
        tsum_loss = 0
        tsum_acc = 0
        # vsum_loss = 0
        # vsum_acc = 0
        pbar = tqdm(range(pretraining_config.BATCH_PER_EPOCH))
        pbar.set_description(f"Epoch {epoch}")
        model.train()
        for i, batch in zip(pbar, train_dataloader):
            images, labels = set_device(batch[1]), set_device(batch[2])
            optimizer.zero_grad()
            outputs = model.clf(model.trunk(images))
            loss = CEL(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            tsum_loss += loss.detach().item()
            _, logits = outputs.max(dim=1)
            tsum_acc += (logits == labels).float().mean()
            pbar.set_postfix({"loss": loss.detach().item()})

        # for batch in pbar_val:
        #     model.eval()
        #     images,labels = set_device(batch[0]),set_device(batch[1])
        #     outputs = classifier(model(images))
        #     v_loss = CEL(outputs,labels-65).detach().item()
        #     _,logits = outputs.max(dim=1)
        #     v_acc = (logits == labels).float().sum()
        #     vsum_acc += v_acc
        #     vsum_loss += v_loss
        #     pbar.set_postfix({"loss":loss.detach().item()})

        logger.info(
            f"epoch {epoch} - training loss: {tsum_loss / pretraining_config.BATCH_PER_EPOCH:.3f}, accuracy: {tsum_acc * 100 / pretraining_config.BATCH_PER_EPOCH:.2f}"
        )
        # logger.info(f"validation loss: {vsum_loss/i}, accuracy: {vsum_acc/val_dataset_size}")

    logger.info("Pretraining finished")
    return model
