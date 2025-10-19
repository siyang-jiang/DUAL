import os
import random
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from configs import (
    dataset_config,
    erm_training_config,
    experiment_config,
    model_config,
    pretraining_config,
    training_config,
)
from src.NTXentLoss import NTXentLoss
from src.utils import LossManager, get_episodic_loader, set_device

from .erm_training_steps import get_n_batches, print_images


class AbstractTrainer:
    def log_experiment_infomation(self):
        logger.info("Model training started.")
        logger.info(
            f"{training_config.N_TARGET} target, {training_config.N_WAY} way - {training_config.N_SOURCE} shot"
        )
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Backbone: {self.backbone_name}")
        logger.info(f"Trainer: {type(self).__name__}")
        logger.info(f"GPU_ID:{experiment_config.GPU_ID}")

    def __init__(self, model, train_loader, val_loader):
        self.dataset_name = (
            dataset_config.DATASET.__name__
            if hasattr(dataset_config.DATASET, "__name__")
            else dataset_config.DATASET.func.__name__
        )
        self.backbone_name = model_config.BACKBONE.__name__
        self.save_dir = experiment_config.ERM_MODEL_DIR
        self.log_experiment_infomation()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.state_dict_path = self.save_dir / f"model.pt"

    def save_model(self, epoch):
        logger.success(f"Best model found at training epoch {epoch}.")
        torch.save(self.model.state_dict(), self.state_dict_path)

    def set_device_batch(self, batch):
        batch = list(batch)
        for i, data in enumerate(batch):
            if isinstance(data, torch.Tensor):
                batch[i] = set_device(data)
            elif isinstance(data, list) or isinstance(data, tuple):
                batch[i] = self.set_device_batch(data)

        return batch

    def train(self):
        raise NotImplementedError

    def training_epoch(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def training_epoch_end(self, *args, **kwarys):
        raise NotImplementedError

    def validation(self):
        raise NotImplementedError


class ermTrainer(AbstractTrainer):  # Original PGADA trainer
    def __init__(self, model, train_loader, val_loader):
        super().__init__(model, train_loader, val_loader)
        logger.info(
            f"REPAIER:{experiment_config.SMART_RESIZER}"
        )  # if we do not train R but inference with R
        self.n_training_batches = get_n_batches(
            self.train_loader, erm_training_config.N_TRAINING_IMAGES_PER_EPOCH
        )
        self.n_val_batches = get_n_batches(
            self.val_loader, erm_training_config.N_VAL_IMAGES_PER_EPOCH
        )
        # self.n_training_batches = 100
        # self.n_val_batches = 100
        if "FULLMeta" in dataset_config.DATASET.__name__:
            self.SIMCLR_loader_train, self.SIMCLR_loader_val = (
                self.get_SIMCLR_loader_meta()
            )
        else:
            self.SIMCLR_loader_train, self.SIMCLR_loader_val = self.get_SIMCLR_loader()
        self.min_val_loss = float("inf")
        self.max_val_acc = -float("inf")
        self.patience = erm_training_config.PATIENCE
        self.loss_dict = ["SimCLR", "Model_CE"]  # remove H_CE, H_COS when fix G
        if not experiment_config.FIX_G:
            self.loss_dict += ["H_CE", "H_COS"]
        self.loss_manager = LossManager(self.loss_dict)

    def get_SIMCLR_loader(self):
        logger.info("Getting SIMCLR loader...")
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
            drop_last=True,
        )
        test_loader_val = DataLoader(
            test_set_SIMCLR_val,
            batch_size=erm_training_config.BATCH_SIZE,
            num_workers=erm_training_config.N_WORKERS,
            shuffle=False,
            drop_last=True,
        )
        if erm_training_config.batch_validate:
            self.model.loss_fn_SIMCLR_val = NTXentLoss(
                f"cuda:{experiment_config.GPU_ID}",
                erm_training_config.BATCH_SIZE,
                erm_training_config.SIMCLR_temp,
                True,
            )
        else:
            self.model.loss_fn_SIMCLR_val = NTXentLoss(
                f"cuda:{experiment_config.GPU_ID}",
                len(test_set_SIMCLR_val),
                erm_training_config.SIMCLR_temp,
                True,
            )
        return test_loader_train, test_loader_val

    def get_SIMCLR_loader_meta(self):
        logger.info("Getting SIMCLR loader of meta dataset...")
        from src.data_tools.datasets.meta_dataset.args import get_args_parser
        from src.data_tools.datasets.meta_dataset import config as config_lib
        from src.data_tools.datasets.meta_dataset.utils import Split
        from src.data_tools.datasets.meta_h5_dataset import FullMetaDatasetH5_ERM
        from src.data_tools.datasets.meta_val_dataset import MetaValDataset

        args = get_args_parser().parse_args()
        args.min_ways = 1
        args.nValEpisode = 120
        args.max_ways_upper_bound = 1
        args.num_support = 1
        args.num_query = 1
        train_dataset = FullMetaDatasetH5_ERM(args, Split.TRAIN, SIMCLR=True)
        val_dataset = FullMetaDatasetH5_ERM(args, Split.VALID, SIMCLR=True)
        test_loader_train = DataLoader(
            train_dataset,
            batch_size=erm_training_config.BATCH_SIZE,
            num_workers=erm_training_config.N_WORKERS,
            shuffle=True,
            drop_last=True,
        )
        test_loader_val = DataLoader(
            val_dataset,
            batch_size=erm_training_config.BATCH_SIZE,
            num_workers=erm_training_config.N_WORKERS,
            shuffle=True,
            drop_last=True,
        )
        self.model.loss_fn_SIMCLR_val = NTXentLoss(
            f"cuda:{experiment_config.GPU_ID}",
            erm_training_config.BATCH_SIZE,
            erm_training_config.SIMCLR_temp,
            True,
        )
        return test_loader_train, test_loader_val

    def train(self):
        logger.info("Model and data are ready. Starting training...")
        best_model_epoch = 0
        for epoch in range(erm_training_config.N_EPOCHS):
            if epoch > best_model_epoch + self.patience:
                logger.info(f"Training early stops.")
                return
            average_loss = self.training_epoch(epoch)
            # wandb.log({"avg loss": average_loss, "epoch": epoch})
            # self.wandb.log("Train/loss", average_loss, epoch)
            # if erm_training_config.SIMCLR:
            #     val_acc, val_loss = self.validation(epoch)
            #     wandb.log({"Val/loss": val_loss, "Val/acc": val_acc, "epoch": epoch})
            #     logger.info(f"epoch {epoch} : Val loss = {val_loss}, Val acc = {val_acc}")
            #     # self.wandb.log("Val/loss", val_loss, epoch)
            #     if val_loss < self.min_val_loss:
            #         self.min_val_loss = val_loss
            #         best_model_epoch = epoch
            #         self.save_model(epoch)
            # else:  # Generally, we do not use this branch
            val_acc, val_loss = self.validation(epoch)
            logger.info(f"epoch {epoch} : Val loss = {val_loss}, Val acc = {val_acc}")
            wandb.log({"Val/acc": val_acc, "epoch": epoch})
            # self.wandb.log("Val/acc", val_acc, epoch)
            if val_acc > self.max_val_acc:
                self.max_val_acc = val_acc
                best_model_epoch = epoch
                self.save_model(epoch)
        logger.success("Training finished")
        return

    def training_epoch(self, epoch):
        wandb_log_inverval = 10
        self.model.train()
        self.loss_manager.reset()
        pbar = tqdm(
            range(self.n_training_batches), desc=f"Epoch {epoch}", dynamic_ncols=True
        )
        for batch_id, train_loader_batch, SimCLR_loader_batch in zip(
            pbar, self.train_loader, self.SIMCLR_loader_train
        ):
            SimCLR_batch = SimCLR_loader_batch[0]
            image_batch, labels = train_loader_batch[:2]
            losses = self.fit(
                *self.set_device_batch([image_batch, SimCLR_batch, labels])
            )
            self.loss_manager.batch_append(losses)
            # loss_outputs = self.loss_manager.get_outputs(type = "last", with_sum=True)
            if batch_id % wandb_log_inverval == 0:
                loss_dicts = self.loss_manager.get_dicts(type="last", with_sum=True)
                loss_dicts = {"Train/Loss/" + k: v for k, v in loss_dicts.items()}
                wandb.log(loss_dicts)
            pbar.set_postfix(self.loss_manager.get_dicts(type="last"))
        loss_outputs = self.loss_manager.get_outputs(type="mean", with_sum=True)
        logger.info(f"epoch {epoch} : {loss_outputs}")
        self.training_epoch_end(epoch, image_batch)
        return self.loss_manager.get_losses_sum(type="mean")

    def fit(self, image_batch, SimCLR_batch, labels):
        images, images_perturbated = image_batch[:2]
        test_img_1, test_img_2 = SimCLR_batch[:2]
        losses = {}
        if not experiment_config.FIX_G:
            losses.update(
                self.fit_H(images, images_perturbated, labels)
            )  # remove this line to fix G
        losses.update(
            self.fit_Model(images, images_perturbated, labels, test_img_1, test_img_2)
        )
        return losses

    def fit_H(self, images, images_perturbation, labels):
        model = self.model
        model.optimizer_H.zero_grad()
        model.H.eval()  # do not use dropout & BN
        f_H = model.trunk(model.H(images))  # H is the generator
        f_perturbation = model.trunk(images_perturbation)
        loss_H_cos_similarity = F.cosine_similarity(f_H, f_perturbation).mean()
        out = model.clf(f_H)
        loss_H_CE = model.loss_fn(out, labels)
        loss_H = loss_H_CE + loss_H_cos_similarity
        loss_H.backward()  # also update paramaters of φ
        model.optimizer_H.step()
        model.H.train()
        return {"H_CE": loss_H_CE.item(), "H_COS": loss_H_cos_similarity.item()}

    def fit_Model(self, images, images_perturbation, labels, test_img_1, test_img_2):
        model = self.model
        model.optimizer.zero_grad()
        with torch.no_grad():  # fix H
            images_H = model.H(images)
        out = model.clf(model.trunk(images_H))
        out_p = model.clf(model.trunk(images_perturbation))
        loss_CE = model.loss_fn(out, labels) + model.loss_fn(out_p, labels)
        if erm_training_config.SIMCLR:
            z1 = model.clf_SIMCLR(model.trunk(test_img_1))
            z2 = model.clf_SIMCLR(model.trunk(test_img_2))
            loss_SIMCLR = model.loss_fn_SIMCLR(z1, z2)
            loss_M_clf = loss_CE + loss_SIMCLR
        else:
            loss_M_clf = loss_CE
            loss_SIMCLR = torch.tensor(0)
        loss_M_clf.backward()
        model.optimizer.step()
        return {"Model_CE": loss_CE.item(), "SimCLR": loss_SIMCLR.item()}

    def validation(self, epoch):
        losses_SIMCLR = []
        val_acc_list = []
        self.model.eval()
        with torch.no_grad():
            # pbar = tqdm(range(self.n_val_batches), desc=f"Validation (SIMCLR)", dynamic_ncols=True)
            # for batch_id, test_loader_batch in zip(pbar, self.test_loader_val):
            #     [test_img1, test_img2] = self.set_device_batch(test_loader_batch[0][:2])
            #     z1 = self.model.clf_SIMCLR(self.model.trunk(test_img1))
            #     z2 = self.model.clf_SIMCLR(self.model.trunk(test_img2))
            #     if erm_training_config.batch_validate:
            #         if len(test_img1) != erm_training_config.BATCH_SIZE:
            #             criterion_small_set = NTXentLoss(f"cuda:{experiment_config.GPU_ID}", len(test_img1), erm_training_config.SIMCLR_temp, True)
            #             losses_SIMCLR.append(criterion_small_set(z1, z2))
            #         else:
            #             losses_SIMCLR.append(self.model.loss_fn_SIMCLR_val(z1, z2))
            # loss_SIMCLR = torch.stack(losses_SIMCLR).mean()
            loss_SIMCLR = torch.tensor(0)
            loss = 0
            pbar = tqdm(
                range(self.n_val_batches), desc=f"Validation", dynamic_ncols=True
            )
            for batch_id, test_loader_batch in zip(pbar, self.val_loader):
                image_batch, labels, _ = self.set_device_batch(test_loader_batch)
                images_perturbation = image_batch[1]
                out_p = self.model.clf(
                    self.model.trunk(images_perturbation)
                )  # QUESTION: should we use R here?
                val_acc_list.append(
                    (torch.argmax(out_p, dim=1) == labels).sum().float() / len(labels)
                )
                loss += self.model.loss_fn(out_p, labels)
        val_acc = torch.stack(val_acc_list).mean()
        # TODO: the FEMNIST version (I HATE FEMNIST)
        return val_acc.item(), loss_SIMCLR.item()

    # pytorch calcuate classification accuracy


class ermTrainer_TFT(AbstractTrainer):  # Pure erm trainer, without PGADA and DuaL
    def __init__(self, model, train_loader, val_loader):
        super().__init__(model, train_loader, val_loader)
        logger.info(
            f"REPAIER:{experiment_config.SMART_RESIZER}"
        )  # if we do not train R but inference with R
        self.n_training_batches = get_n_batches(
            self.train_loader, erm_training_config.N_TRAINING_IMAGES_PER_EPOCH
        )
        self.n_val_batches = get_n_batches(
            self.val_loader, erm_training_config.N_VAL_IMAGES_PER_EPOCH
        )
        self.patience = erm_training_config.PATIENCE
        self.loss_dict = ["Model_CE"]
        self.loss_manager = LossManager(self.loss_dict)
        self.max_val_acc = -float("inf")

    def train(self):
        logger.info("Model and data are ready. Starting training...")
        best_model_epoch = 0
        for epoch in range(erm_training_config.N_EPOCHS):
            if epoch > best_model_epoch + self.patience:
                logger.info(f"Training early stops.")
                return
            self.training_epoch(epoch)
            val_acc = self.validation(epoch)
            wandb.log({"Val/acc": val_acc, "epoch": epoch})
            logger.info(f"Val acc: {val_acc:.4f}, best val acc: {self.max_val_acc:.4f}")
            if val_acc > self.max_val_acc:
                self.max_val_acc = val_acc
                best_model_epoch = epoch
                self.save_model(epoch)
        logger.success("Training finished")
        return

    def training_epoch(self, epoch):
        wandb_log_inverval = 10
        self.model.train()
        self.loss_manager.reset()
        pbar = tqdm(
            range(self.n_training_batches), desc=f"Epoch {epoch}", dynamic_ncols=True
        )
        for batch_id, train_loader_batch in zip(pbar, self.train_loader):
            image_batch, labels, _ = train_loader_batch
            losses = self.fit(*self.set_device_batch([image_batch, labels]))
            self.loss_manager.batch_append(losses)
            if batch_id % wandb_log_inverval == 0:
                loss_dicts = self.loss_manager.get_dicts(type="last", with_sum=True)
                loss_dicts = {"Train/Loss/" + k: v for k, v in loss_dicts.items()}
                wandb.log(loss_dicts)
            pbar.set_postfix(self.loss_manager.get_dicts(type="last"))
        loss_outputs = self.loss_manager.get_outputs(type="mean", with_sum=True)
        logger.info(f"epoch {epoch} : {loss_outputs}")
        return self.loss_manager.get_losses_sum(type="mean")

    def fit(self, image_batch, labels):
        images = image_batch[0]
        losses = {}
        losses.update(self.fit_Model(images, labels))
        return losses

    def fit_Model(self, images, labels):
        model = self.model
        model.optimizer.zero_grad()
        out = model.clf(model.trunk(images))
        loss_CE = model.loss_fn(out, labels)
        loss_CE.backward()
        model.optimizer.step()
        return {"Model_CE": loss_CE.item()}

    def validation(self, epoch):
        val_acc_list = []
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(
                range(self.n_val_batches), desc=f"Validation", dynamic_ncols=True
            )
            for batch_id, test_loader_batch in zip(pbar, self.val_loader):
                image_batch, labels, _ = self.set_device_batch(test_loader_batch)
                images = image_batch[0]
                out_p = self.model.clf(
                    self.model.trunk(images)
                )  # QUESTION: should we use R here?
                val_acc_list.append(
                    (torch.argmax(out_p, dim=1) == labels).sum().float() / len(labels)
                )
        val_acc = torch.stack(val_acc_list).mean()
        return val_acc.item()


class ermTrainerResize(ermTrainer):
    def _onebyone_downsapmling(self, images, input_function):
        return torch.stack([input_function(image) for image in images])

    def __init__(self, model, train_loader, val_loader):
        super().__init__(model, train_loader, val_loader)
        logger.info(f"Resizer Type:{model_config.R.__name__}")
        self.loss_dict.append("Resizer")
        if experiment_config.CELOSS_R:
            self.loss_dict.append("loss_CE_R")
        self.loss_manager = LossManager(self.loss_dict)

    def set_R(self, R):
        self.model.R = R

    def save_fig(self, image, name):
        temp = TF.to_pil_image(image)
        temp.save(f"{name}.jpg")

    def fit(self, image_batch, SimCLR_batch, labels):
        images, images_pertubated = image_batch[:2]
        test_img_1, test_img_2, test_img_pure, test_img_p = SimCLR_batch
        # kwargs = {"pertubated_images": images_pertubated_2, "target_images": images}
        kwargs = {"pertubated_images": test_img_p, "target_images": test_img_pure}
        if experiment_config.CELOSS_R_IN_R:
            kwargs.update({"labels": labels})
        losses = self.fit_R(**kwargs)
        losses.update(super().fit([images, images_pertubated], SimCLR_batch, labels))
        return losses

    def fit_R(self, pertubated_images, target_images, labels=None):
        model = self.model
        model.optimizer_R.zero_grad()
        f_R = model.trunk(model.R(pertubated_images))
        with torch.no_grad():
            f_target = model.trunk(target_images).detach()
        loss_R = (
            F.kl_div(F.log_softmax(f_R), F.softmax(f_target))
            if experiment_config.R_KLD
            else -F.cosine_similarity(f_R, f_target).mean()
        )

        if experiment_config.CELOSS_R_IN_R:
            out_r = model.clf(f_R)
            loss_CE = model.loss_fn(out_r, labels)
            loss_R = loss_R + loss_CE

        loss_R.backward()
        model.optimizer_R.step()
        return {"Resizer": loss_R}

    def fit_Model(self, images, images_perturbation, labels, test_img_1, test_img_2):
        model = self.model
        model.optimizer.zero_grad()
        with torch.no_grad():  # fix H
            images_H = model.H(images)
        out = model.clf(model.trunk(images_H))
        out_p = (
            model.clf(model.trunk(images_perturbation))
            if not experiment_config.ENCODING_IN_PHI
            else model.clf(model.trunk(model.R(images_perturbation)))
        )

        if experiment_config.CELOSS_R:
            out_r = model.clf(model.trunk(model.R(images)))
            loss_r = model.loss_fn(out_r, labels)
            loss_CE = model.loss_fn(out, labels) + model.loss_fn(out_p, labels) + loss_r
        else:
            loss_CE = (
                model.loss_fn(out, labels) * 0.9 + model.loss_fn(out_p, labels) * 0.1
            )

        if erm_training_config.SIMCLR:
            if test_img_1.shape[0] == erm_training_config.BATCH_SIZE:
                z1 = model.clf_SIMCLR(model.trunk(test_img_1))
                z2 = model.clf_SIMCLR(model.trunk(test_img_2))
                loss_SIMCLR = model.loss_fn_SIMCLR(z1, z2)
                loss_M_clf = loss_CE + loss_SIMCLR
        else:
            loss_M_clf = loss_CE
            loss_SIMCLR = torch.tensor(0)

        loss_M_clf.backward()
        model.optimizer.step()
        losses = {"Model_CE": loss_CE.item(), "SimCLR": loss_SIMCLR.item()}
        if experiment_config.CELOSS_R:
            losses.update({"loss_CE_R": loss_r})
        return losses

    # def train_R(self, max_epoch):  # For training R only, not be used in erm training, Unavailbable for now
    #     logger.info("Training R")
    #     for epoch in range(max_epoch):
    #         self.model.train()
    #         self.loss_manager.reset()
    #         pbar = tqdm(range(self.n_training_batches), desc=f"Epoch {epoch}", dynamic_ncols=True)
    #         for batch_id, train_loader_batch, SimCLR_loader_batch in zip(pbar, self.train_loader, self.test_loader_train):
    #             test_img1, test_img2, _, _ = SimCLR_loader_batch
    #             if self.dataset_name == "FEMNIST":
    #                 support_images, support_labels, query_images, query_labels, class_ids, source_domain, target_domain = train_loader_batch
    #                 labels = torch.as_tensor([class_ids[i] for i in support_labels], dtype=torch.long)
    #                 losses = self.fit_R(set_device(support_images))
    #             else:
    #                 images, images_p, _, _ = train_loader_batch
    #                 losses = self.fit_R(images=set_device(images), target_images=set_device(images))

    #             self.loss_manager.batch_append(losses)
    #             pbar.set_postfix(self.loss_manager.get_dicts(loss_list=["Resizer"], type="last"))
    #         loss_outputs = self.loss_manager.get_outputs(loss_list=["Resizer"], type="mean")
    #         logger.info(f"R training epoch {epoch} : {loss_outputs}")
    #         self.training_epoch_end(epoch, test_img1)

    def training_epoch_end(self, epoch, image_batch):
        # for smart resizer
        figure_path = self.save_dir / "figures"
        if not figure_path.exists():
            os.makedirs(figure_path)
        images, images_p = image_batch[:2]
        self.model.R.eval()
        self.model.eval()
        image_list, similarity_R, similarity_baseline, improvement = self.print_ready(
            images, images_p
        )
        titles = ["Original", "Small", "Resize", "Learnable Resizer"]
        # print_images(image_list, titles, self.save_dir / "figures" / f"epoch_{epoch}_images")
        image_list.pop(2)
        examples = []
        image_log = [
            wandb.Image(image_list[i], caption=name)
            for i, name in enumerate(["Original", "Pertubation", "Repaired"])
        ]
        wandb.log({"img": image_log})
        logger.info(
            f"epoch {epoch} : R Similarity = {similarity_R:.3f}, Bilinear Similarity = {similarity_baseline:.3f}, R improvement = {improvement:.3f}"
        )

    def print_test_images(self, test_loader):
        for i in range(10):
            batch = next(iter(test_loader))
            self.print_resized_testing_images(batch[0], f"test_support_image_{i}")
            self.print_resized_testing_images(batch[2], f"test_query_image_{i}")

    def print_ready(self, images, images_p2, image_id=0):
        with torch.no_grad():
            images = set_device(images)
            small_images = set_device(images_p2)
            upsampled_image = self.model.R(small_images)
            f_R = self.model.trunk(upsampled_image)
            f = self.model.trunk(images)
            similarity_baseline = F.cosine_similarity(
                self.model.trunk(small_images), f
            ).mean()
            similarity_R = F.cosine_similarity(f_R, f).mean()
            improvement = similarity_R - similarity_baseline
            image_list = [
                images[0],
                small_images[0],
                small_images[0],
                upsampled_image[0],
            ]
        return image_list, similarity_R, similarity_baseline, improvement

    def print_resized_testing_images(self, images, filename):
        image_list, similarity_R, similarity_baseline, improvement = self.print_ready(
            images, images
        )
        titles = ["Original", "Small", "Resize", "Learnable Resizer"]
        print_images(image_list, titles, self.save_dir / "figures" / f"{filename}")
        logger.info(
            f"[{filename}] R Similarity = {similarity_R:.3f}, Resize Similarity = {similarity_baseline:.3f}, R improvement = {improvement:.3f}"
        )


"""
from torchvision.transforms import functional as tff


class ermTrainerResizeVariable(ermTrainerResize):
    def downsample_function(self, image):
        # level = self.downsampling_level
        level = random.choice(self.ds_level_list)
        image = self.old_downsampling(tff.resize(image, (image.shape[2] // level)))
        return image

    def __init__(self, model, train_loader, val_loader):
        super().__init__(model, train_loader, val_loader)
        self.old_downsampling = self.downsampling
        self.downsampling = self.downsample_function
        self.ds_level_list = [2, 4, 8]


class supervisedTrainerResize(ermTrainerResize):
    def __init__(self, model, train_loader, val_loader):
        super().__init__(model, train_loader, val_loader)
        self.loss_dict = ["loss", "acc"]
        self.loss_manager = LossManager(self.loss_dict)

    def train(self):
        logger.info("Model and data are ready. Starting training...")
        best_model_epoch = 0
        for epoch in range(erm_training_config.N_EPOCHS):
            average_loss = self.training_epoch(epoch)
            self.validation(epoch)
        return

    def training_epoch_end(self, *args, **kwargs):
        return

    def fit(self, images, images_perturbation, labels, test_img_1, test_img_2):
        model = self.model
        model.optimizer.zero_grad()
        images = self.downsampling(images)
        out = model.clf(model.trunk(images))
        loss_CE = model.loss_fn(out, labels)
        acc = (out.argmax(dim=1) == labels).float().mean()
        loss_CE.backward()
        model.optimizer.step()
        losses = {"loss": loss_CE, "acc": acc}
        return losses

    def eval(self, images, images_perturbation, labels, test_img_1, test_img_2):
        with torch.no_grad():
            model = self.model
            out = model.clf(model.trunk(images))
            loss_CE = model.loss_fn(out, labels)
            acc = (out.argmax(dim=1) == labels).float().mean()
            losses = {"loss": loss_CE, "acc": acc}
        return losses

    def validation(self, epoch):
        self.model.eval()
        self.loss_manager.reset()
        pbar = tqdm(range(self.n_val_batches), desc=f"Validation", dynamic_ncols=True)
        for batch_id, val_loader_batch, SimCLR_loader_batch in zip(pbar, self.val_loader, self.test_loader_val):
            test_img1, test_img2, _, _, _ = SimCLR_loader_batch
            if self.dataset_name == "FEMNIST":
                support_images, support_labels, query_images, query_labels, class_ids, source_domain, target_domain = val_loader_batch
                labels = torch.as_tensor([class_ids[i] for i in support_labels], dtype=torch.long)
                losses = self.eval(*self.set_device_batch((support_images, query_images, labels, test_img1, test_img2)))
            else:
                images, images_perturbation, query_images, labels, _ = val_loader_batch
                losses = self.eval(*self.set_device_batch((images, images_perturbation, labels, test_img1, test_img2)))
            self.loss_manager.batch_append(losses)
            loss_outputs = self.loss_manager.get_outputs(type="last")
            pbar.set_postfix(self.loss_manager.get_dicts(type="last"))
        loss_outputs = self.loss_manager.get_outputs(type="mean")
        logger.info(f"Validation : {loss_outputs}")

"""
