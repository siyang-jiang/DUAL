"""
Steps used in scripts/erm_training.py
"""

# from typing import OrderedDict
from multiprocessing import reduction
from matplotlib import pyplot as plt
from collections import OrderedDict

# from src.modules import backbones

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import optimizer
from torch.utils.data import ConcatDataset, random_split, DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import (
    dataset_config,
    erm_training_config,
    experiment_config,
    model_config,
    training_config,
    pretraining_config,
)
from src.utils import set_device, get_episodic_loader
import configs.evaluation_config
from src.NTXentLoss import NTXentLoss
import copy
from torchvision import transforms

# resizer = transforms.Resize(int(dataset_config.IMAGE_SIZE / experiment_config.TESTING_QUERY_RESIZE_FACTOR))
# inv_resizer = transforms.Resize(dataset_config.IMAGE_SIZE)


class projector_SIMCLR(nn.Module):
    """
    The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    """

    def __init__(self, in_dim, out_dim):
        super(projector_SIMCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class LearnableResizer(nn.Module):
    def __init__(self):
        super(LearnableResizer, self).__init__()
        self.core = model_config.R()

    def forward(self, x):
        # code = None
        x = self.core(x)
        # x,code = self.core(x) # Auto encoder
        # if torch.max(x)>1: # If the image output is not in [0,1]. Not sure should I do it here or in the training stage.
        #     x = x/torch.max(x)
        return x


def get_few_shot_split(two_stream=False) -> tuple([Dataset, Dataset]):
    temp_train_set = dataset_config.DATASET(
        dataset_config.DATA_ROOT,
        "train",
        dataset_config.IMAGE_SIZE,
        two_stream=two_stream,
    )
    temp_train_classes = len(temp_train_set.id_to_class)
    temp_val_set = dataset_config.DATASET(
        dataset_config.DATA_ROOT,
        "val",
        dataset_config.IMAGE_SIZE,
        target_transform=lambda label: label + temp_train_classes,
        two_stream=two_stream,
    )
    if hasattr(dataset_config.DATASET, "__name__"):
        # transform label by id_to_class (why ?)
        if dataset_config.DATASET.__name__ == "CIFAR100CMeta":
            label_mapping = {
                v: k
                for k, v in enumerate(
                    list(temp_train_set.id_to_class.keys())
                    + list(temp_val_set.id_to_class.keys())
                )
            }
            temp_train_set.target_transform = temp_val_set.target_transform = (
                lambda label: label_mapping[label]
            )

    return temp_train_set, temp_val_set


def get_non_few_shot_split(temp_train_set: Dataset, temp_val_set: Dataset) -> tuple(
    [Subset, Subset]
):
    train_and_val_set = ConcatDataset(
        [
            temp_train_set,
            temp_val_set,
        ]
    )
    n_train_images = int(
        len(train_and_val_set) * erm_training_config.TRAIN_IMAGES_PROPORTION
    )
    return random_split(
        train_and_val_set,
        [n_train_images, len(train_and_val_set) - n_train_images],
        generator=torch.Generator().manual_seed(
            erm_training_config.TRAIN_VAL_SPLIT_RANDOM_SEED
        ),
    )


# mix training and validation


def get_data(two_stream=False) -> tuple([DataLoader, DataLoader, int]):
    logger.info("Initializing data loaders...")

    if "MetaDataset" in dataset_config.DATASET.__name__:
        from src.data_tools.datasets.meta_h5_dataset import (
            FullMetaDatasetH5,
            FullMetaDatasetH5_ERM,
        )
        from src.data_tools.datasets.meta_dataset.args import get_args_parser
        from src.data_tools.datasets.meta_dataset.utils import Split

        args = get_args_parser().parse_args()
        args.base_sources = ["ilsvrc_2012"]
        args.image_size = 128
        args.min_ways = 1
        args.max_ways_upper_bound = 1
        args.num_support = 1
        args.num_query = 1

        if two_stream:
            train_set = FullMetaDatasetH5_ERM(args, Split.TRAIN)
            val_set = FullMetaDatasetH5_ERM(args, Split.VALID)
            train_loader = DataLoader(
                train_set,
                batch_size=erm_training_config.BATCH_SIZE,
                num_workers=erm_training_config.N_WORKERS,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=erm_training_config.BATCH_SIZE,
                num_workers=erm_training_config.N_WORKERS,
            )
            n_classes = train_set.num_classes
    else:
        temp_train_set, temp_val_set = get_few_shot_split(two_stream)

        train_set, val_set = get_non_few_shot_split(temp_train_set, temp_val_set)

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
        # Assume that train and val classes are entirely disjoints
        n_classes = len(temp_val_set.id_to_class) + len(temp_train_set.id_to_class)

    return train_loader, val_loader, n_classes


def get_model(n_classes: int) -> nn.Module:
    logger.info(f"Initializing {model_config.BACKBONE.__name__}...")

    model = set_device(model_config.BACKBONE())

    model.clf = set_device(nn.Linear(model.final_feat_dim, n_classes))
    model.H = set_device(model_config.H())
    # model.H = set_device(nn.Identity())  # ! TEMP
    model.clf_SIMCLR = set_device(
        projector_SIMCLR(
            model.final_feat_dim, erm_training_config.SIMCLR_projection_dim
        )
    )
    model.R = set_device(LearnableResizer())
    model.softmax = nn.Softmax(dim=1)
    model.loss_NLL = nn.NLLLoss()
    model.loss_fn = nn.CrossEntropyLoss(reduction="mean")  # SY ADD
    model.loss_fn_SIMCLR = NTXentLoss(
        f"cuda:{experiment_config.GPU_ID}",
        erm_training_config.BATCH_SIZE,
        erm_training_config.SIMCLR_temp,
        True,
    )

    model.optimizer = erm_training_config.OPTIMIZER(
        list(model.trunk.parameters())
        + list(model.clf.parameters())
        + list(model.clf_SIMCLR.parameters())
    )
    model.optimizer_H = erm_training_config.OPTIMIZER(model.H.parameters())
    model.optimizer_R = erm_training_config.OPTIMIZER(model.R.parameters())
    return model


def get_n_batches(data_loader: DataLoader, n_images_per_epoch: int) -> int:
    """
    Computes the number of batches in a training epoch from the intended number of seen images.
    """

    return min(n_images_per_epoch // erm_training_config.BATCH_SIZE, len(data_loader))


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    persudo_model=None,
) -> tuple([OrderedDict, int]):
    logger.info("Model training started.")
    logger.info(
        f"{training_config.N_TARGET} target, {training_config.N_WAY} way - {training_config.N_SOURCE} shot"
    )
    logger.info(f"Backbone: {model_config.BACKBONE}")
    logger.info(f"Model: {model_config.MODEL}")
    logger.info(f"Seed: {experiment_config.RANDOM_SEED}")
    logger.info(f"Resize factor: {experiment_config.TESTING_QUERY_RESIZE_FACTOR}")
    logger.info("Initializing data loaders...")
    persudo_model = set_device(persudo_model) if persudo_model else None
    writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)
    n_training_batches = get_n_batches(
        train_loader, erm_training_config.N_TRAINING_IMAGES_PER_EPOCH
    )
    n_val_batches = get_n_batches(
        val_loader, erm_training_config.N_VAL_IMAGES_PER_EPOCH
    )

    test_set_temp1 = dataset_config.DATASET(
        dataset_config.DATA_ROOT,
        "test",
        dataset_config.IMAGE_SIZE,
        SIMCLR=True,
        two_stream=True,
    )
    test_set_SIMCLR_val = dataset_config.DATASET(
        dataset_config.DATA_ROOT,
        "test",
        dataset_config.IMAGE_SIZE,
        SIMCLR=True,
        two_stream=True,
        SIMCLR_val=True,
    )

    ind = torch.randperm(len(test_set_temp1))

    test_set_train_ind = ind[: int(0.9 * len(ind))]
    test_set_val_ind = ind[int(0.9 * len(ind)) :]

    test_set_train = Subset(test_set_temp1, test_set_train_ind)
    test_set_SIMCLR_val = Subset(test_set_SIMCLR_val, test_set_val_ind)

    test_loader_train = DataLoader(
        test_set_train,
        batch_size=erm_training_config.BATCH_SIZE,
        num_workers=erm_training_config.N_WORKERS,
        shuffle=True,
    )

    test_loader_val = DataLoader(
        test_set_SIMCLR_val,
        batch_size=erm_training_config.BATCH_SIZE,
        num_workers=erm_training_config.N_WORKERS,
        shuffle=False,
    )

    if erm_training_config.batch_validate:
        model.loss_fn_SIMCLR_val = NTXentLoss(
            f"cuda:{experiment_config.GPU_ID}",
            erm_training_config.BATCH_SIZE,
            erm_training_config.SIMCLR_temp,
            True,
        )
    else:
        model.loss_fn_SIMCLR_val = NTXentLoss(
            f"cuda:{experiment_config.GPU_ID}",
            len(test_set_SIMCLR_val),
            erm_training_config.SIMCLR_temp,
            True,
        )

    if erm_training_config.SIMCLR:
        min_val_loss = float("inf")
    else:
        max_val_acc = -float("inf")
    best_model_epoch = 0
    logger.info("Model and data are ready. Starting training...")
    for epoch in range(erm_training_config.N_EPOCHS):
        if epoch > best_model_epoch + 10:
            logger.info(f"Training early stops.")
            return

        model, average_loss = training_epoch(
            model,
            train_loader,
            epoch,
            n_training_batches,
            test_loader_train,
            persudo_model,
        )

        writer.add_scalar("Train/loss", average_loss, epoch)

        if erm_training_config.SIMCLR:
            val_loss = validation(
                model, val_loader, epoch, n_val_batches, test_loader_val
            )

            writer.add_scalar("Val/loss", val_loss, epoch)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model_epoch = epoch
                logger.success(
                    f"Best model found at training epoch {best_model_epoch}."
                )
                state_dict_path = (
                    experiment_config.SAVE_DIR
                    / f"{model_config.BACKBONE.__name__}_{dataset_config.DATASET.__name__ if hasattr(dataset_config.DATASET, '__name__') else dataset_config.DATASET.func.__name__}_{epoch}.tar"
                )
                torch.save(model.state_dict(), state_dict_path)
        else:
            val_acc = validation_acc(
                model, val_loader, epoch, n_val_batches, test_loader_val
            )
            writer.add_scalar("Val/acc", val_acc, epoch)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_model_epoch = epoch
                logger.success(
                    f"Best model found at training epoch {best_model_epoch} ."
                )
                state_dict_path = (
                    experiment_config.SAVE_DIR
                    / f"{model_config.BACKBONE.__name__}_{dataset_config.DATASET.__name__ if hasattr(dataset_config.DATASET, '__name__') else dataset_config.DATASET.func.__name__}_{epoch}.tar"
                )
                torch.save(model.state_dict(), state_dict_path)
    return


def training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    epoch: int,
    n_batches: int,
    test_loader_train: DataLoader,
    persudo_model=None,
) -> tuple([nn.Module, float]):
    loss_clf_list = []
    loss_cos_list = []
    loss_R_list = []
    model.train()

    if dataset_config.DATASET.__name__ == "FEMNIST":
        for (
            batch_id,
            (
                support_images,
                support_labels,
                query_images,
                query_labels,
                class_ids,
                source_domain,
                target_domain,
            ),
            (test_img1, test_img2, _, _),
        ) in zip(range(n_batches), data_loader, test_loader_train):
            labels = torch.as_tensor(
                [class_ids[i] for i in support_labels], dtype=torch.long
            )
            model, loss_clf, loss_cos = fit(
                model,
                set_device(support_images),
                set_device(query_images),
                set_device(labels),
                set_device(test_img1),
                set_device(test_img2),
            )  # meta-training # test_img?

            loss_clf_list.append(loss_clf)
            loss_cos_list.append(loss_cos)

            print(
                f"epoch {epoch} [{batch_id + 1:03d}/{n_batches}]: clf loss={np.asarray(loss_clf_list).mean():.3f}, cos loss={np.asarray(loss_cos_list).mean():.3f}",
                end="     \r",
            )
    else:
        for (
            batch_id,
            (images, images_perturbation, labels, _),
            (test_img1, test_img2, _, _),
        ) in zip(range(n_batches), data_loader, test_loader_train):
            model, loss_clf, loss_cos, loss_R = fit(
                model,
                set_device(images),
                set_device(images_perturbation),
                set_device(labels),
                set_device(test_img1),
                set_device(test_img2),
                persudo_model,
            )

            loss_clf_list.append(loss_clf)
            loss_cos_list.append(loss_cos)
            loss_R_list.append(loss_R)

            print(
                (
                    f"epoch {epoch} [{batch_id + 1:04d}/{n_batches}]: clf loss={np.asarray(loss_clf_list).mean():.3f}, cos loss={np.asarray(loss_cos_list).mean():.3f}, loss_dit"
                    f" ={np.asarray(loss_R_list).mean():.3f}"
                ),
                end="     \r",
            )

    if configs.experiment_config.SMART_RESIZER:
        with torch.no_grad():
            images = set_device(images)
            small_images = resizer(images)
            reductioned_image = model.R(small_images)
            f_R = model.trunk(reductioned_image)
            f = model.trunk(images)
            similarity_baseline = F.cosine_similarity(
                model.trunk(inv_resizer(small_images)), f
            ).mean()
            similarity_R = F.cosine_similarity(f_R, f).mean()
            improvement = similarity_R - similarity_baseline
            image_list = [
                images[0],
                small_images[0],
                inv_resizer(small_images)[0],
                reductioned_image[0],
            ]

            titles = ["Original", "Small", "Resize", "Learnable Resizer"]
            print_images(
                image_list, titles, experiment_config.SAVE_DIR / f"epoch_{epoch}_images"
            )

            logger.info(
                f"epoch {epoch}: R Similarity = {similarity_R:.3f}, Resize Similarity = {similarity_baseline:.3f}, R improvement = {improvement:.3f}"
            )

    logger.info(
        f"epoch {epoch} [{batch_id + 1:04d}/{n_batches}]: clf loss={np.asarray(loss_clf_list).mean():.3f}, cos loss={np.asarray(loss_cos_list).mean():.3f}, loss_dit"
        f" ={np.asarray(loss_R_list).mean():.3f}"
    )

    return model, np.asarray(loss_clf_list).mean() + np.asarray(loss_cos_list).mean()


def print_image(img, title=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
    if title is not None:
        plt.title(title)
    plt.show()
    plt.savefig(f"{title}.png")


def print_images(images, titles=None, filename=None):
    fig, axs = plt.subplots(1, len(titles), figsize=(15, 6))
    for i, img in enumerate(images):
        axs[i].imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
        axs[i].set_title(titles[i])
    plt.show()
    plt.savefig(f"{filename}.png")


def fit(
    model: nn.Module,
    images: torch.Tensor,
    images_perturbation: torch.Tensor,
    labels: torch.Tensor,
    test_img1: torch.Tensor,
    test_img2: torch.Tensor,
    persudo_model=None,
) -> tuple([nn.Module, float]):
    if configs.experiment_config.SMART_RESIZER:
        # train R
        model.optimizer_R.zero_grad()
        small_images = resizer(images)
        reductioned_image = model.R(small_images)
        f_R = model.trunk(reductioned_image)
        f = model.trunk(images)
        loss_R = -F.cosine_similarity(f_R, f).mean()
        loss_R.backward()
        model.optimizer_R.step()
    else:
        loss_R = torch.zeros(1)

    # train H (G in paper)
    model.optimizer_H.zero_grad()
    model.H.eval()  # do not use dropout & BN
    f_H = model.trunk(model.H(images))  # H is the generator ?
    f_perturbation = model.trunk(images_perturbation)
    # f_H_norm = f_H / f_H.norm(dim=1)[:, None]
    # f_perturbation_norm = f_perturbation / f_perturbation.norm(dim=1)[:, None]
    # loss_cos_similarity = torch.mm(f_H_norm, f_perturbation_norm.transpose(0,1)).mean()
    loss_cos_similarity = F.cosine_similarity(f_H, f_perturbation).mean()
    out = model.clf(f_H)
    loss_clf = model.loss_fn(out, labels)
    loss_H = loss_clf + loss_cos_similarity
    # loss_H = loss_cos_similarity
    loss_H.backward()  # also update paramaters of φ
    model.optimizer_H.step()

    # train φ & cls
    model.optimizer.zero_grad()
    model.H.train()
    with torch.no_grad():  # fix H
        images_H = model.H(images)
        # images_R = model.R(small_images)
    out = model.clf(model.trunk(images_H))
    # out_r = model.clf(model.trunk(images_R))
    out_p = model.clf(model.trunk(images_perturbation))
    loss_CE = model.loss_fn(out, labels) + model.loss_fn(
        out_p, labels
    )  # +model.loss_fn(out_r, labels)
    if erm_training_config.SIMCLR:
        z1 = model.clf_SIMCLR(model.trunk(test_img1))
        z2 = model.clf_SIMCLR(model.trunk(test_img2))
        # imgs = torch.cat([test_img1, test_img2], dim=0)
        if persudo_model is not None:
            persudo_output = persudo_model.clf_SIMCLR(persudo_model.trunk(test_img1))
            # persudo_label = persudo_output.argmax(dim=1) # hard label
            persudo_label = (
                (persudo_output / pretraining_config.SHARPEN_FACTOR)
                .softmax(dim=1)
                .detach()
            )  # sharpen factor
            predicted_label = model.clf_SIMCLR(model.trunk(test_img2))
            if pretraining_config.KLD:
                loss_dit = F.kl_div(F.log_softmax(predicted_label), persudo_label)
            else:
                loss_dit = cross_entropy(predicted_label, persudo_label)
        else:
            loss_dit = torch.zeros(1)

        # z3 = model.clf_SIMCLR(model.trunk(images_perturbation))
        # z4 = model.clf_SIMCLR(model.trunk(images))

        loss_SIMCLR = model.loss_fn_SIMCLR(z1, z2)
        # loss_SIMCLR2 = model.loss_fn_SIMCLR(z3, z4)
        loss_M_clf = (
            loss_CE + loss_SIMCLR + loss_dit if persudo_model else loss_CE + loss_SIMCLR
        )
        # loss_M_clf = loss_CE  + loss_dit
        if pretraining_config.UPDATE_TEACHER:
            update_model(persudo_model, model)
    else:
        loss_M_clf = loss_CE
    loss_M_clf.backward()  # classifier loss
    model.optimizer.step()

    return model, loss_M_clf.item(), loss_cos_similarity.item(), loss_dit.item()


def update_model(teacher, student):
    m = pretraining_config.TEACHER_MOMENTUM
    for teacher_param, student_param in zip(
        teacher.trunk.parameters(), student.trunk.parameters()
    ):
        teacher_param.data.mul_(m).add_((1 - m) * student_param.detach().data)
        # teacher_param.data.copy_(m + (1.0-m)*student_param.data)

    for teacher_param, student_param in zip(
        teacher.clf.parameters(), student.clf.parameters()
    ):
        teacher_param.data.mul_(m).add_((1 - m) * student_param.detach().data)


def cross_entropy(logits, y_gt):
    if len(y_gt.shape) < len(logits.shape):
        return F.cross_entropy(logits, y_gt, reduction="mean")
    else:
        return (-y_gt * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def validation(
    model: nn.Module,
    data_loader: DataLoader,
    epoch: int,
    n_batches: int,
    test_loader_val: DataLoader,
) -> float:
    if erm_training_config.batch_validate:
        losses_SIMCLR = []
    else:
        z1s = []
        z2s = []

    model.eval()
    with torch.no_grad():
        for batch_id, (test_img1, test_img2, _, _) in zip(
            range(n_batches), test_loader_val
        ):
            test_img1 = set_device(test_img1)
            test_img2 = set_device(test_img2)

            z1 = model.clf_SIMCLR(model.trunk(test_img1))
            z2 = model.clf_SIMCLR(model.trunk(test_img2))

            if erm_training_config.batch_validate:
                if len(test_img1) != erm_training_config.BATCH_SIZE:
                    criterion_small_set = NTXentLoss(
                        f"cuda:{experiment_config.GPU_ID}",
                        len(test_img1),
                        erm_training_config.SIMCLR_temp,
                        True,
                    )
                    losses_SIMCLR.append(criterion_small_set(z1, z2))
                else:
                    losses_SIMCLR.append(model.loss_fn_SIMCLR_val(z1, z2))
            else:
                z1s.append(z1)
                z2s.append(z2)

    if erm_training_config.batch_validate:
        loss_SIMCLR = torch.stack(losses_SIMCLR).mean()
    else:
        z1s = torch.cat(z1s, dim=0)
        z2s = torch.cat(z2s, dim=0)
        loss_SIMCLR = model.loss_fn_SIMCLR_val(z1s, z2s)

    if dataset_config.DATASET.__name__ == "FEMNIST":  # The FEMNIST is episodic dataset
        val_model = set_device(model_config.MODEL(model_config.BACKBONE))
        val_model.feature = model
        val_model.eval()
        with torch.no_grad():
            loss, acc, stats_df = val_model.eval_loop(data_loader)
    else:
        with torch.no_grad():
            for batch_id, (images, images_perturbation, labels, _) in zip(
                range(n_batches), data_loader
            ):
                images_perturbation = set_device(images_perturbation)
                labels = set_device(labels)

                out_p = model.clf(model.trunk(images_perturbation))
                loss = model.loss_fn(out_p, labels)

    logger.info(f"epoch {epoch} : validation loss = {loss + loss_SIMCLR:.3f}")
    return loss + loss_SIMCLR


def validation_acc(
    model: nn.Module,
    data_loader: DataLoader,
    epoch: int,
    n_batches: int,
    test_loader_val: DataLoader,
) -> float:
    if dataset_config.DATASET.__name__ == "FEMNIST":
        val_model = set_device(model_config.MODEL(model_config.BACKBONE))
        val_model.feature = model
        val_model.eval()
        loss, acc, stats_df = val_model.eval_loop(data_loader)
        return acc
    else:
        val_acc_list = []
        model.eval()
        with torch.no_grad():
            for batch_id, (images, images_perturbation, labels, _) in zip(
                range(n_batches), data_loader
            ):
                val_acc_list.append(
                    (
                        model.clf(
                            model.trunk(set_device(images_perturbation))
                        ).data.topk(1, 1, True, True)[1][:, 0]
                        == set_device(labels)
                    )
                    .sum()
                    .float()
                    / len(labels)
                )
                # logger.info(
                #     f"validation [{batch_id+1:03d}/{n_batches}]: acc={np.asarray(val_acc_list).mean():.3f}", end="     \r")
        return np.asarray(val_acc_list).mean()
