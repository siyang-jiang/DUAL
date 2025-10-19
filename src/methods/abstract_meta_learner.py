from abc import abstractmethod

from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from src.methods.utils import confidence_interval
from src.utils import set_device

from configs.experiment_config import INVERSE_RESIZE, SMART_RESIZER
from configs.dataset_config import IMAGE_SIZE


class AbstractMetaLearner(nn.Module):
    """
    Abstract class for meta-learning models. Extensions of this class must define the set_forward function.
    """

    def __init__(self, model_func, transportation=None, training_stats=None):
        """

        Args:
            model_func (src.backbones object): backbone function
            training_stats (Statistics): training statistics of the model, updated during training

        """
        super(AbstractMetaLearner, self).__init__()
        self.feature = model_func()
        self.training_stats = training_stats
        self.loss_fn = nn.CrossEntropyLoss()
        self.transportation_module = transportation

    @abstractmethod
    def set_forward(self, support_images, support_labels, query_images, infer=False):
        """
        Predict query set labels using information from support set labelled images.
        Must be implemented in all classes extending AbstractMetaLearner.
        Args:
            support_images (torch.Tensor): shape (number_of_support_set_images, **image_shape)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
            query_images (torch.Tensor): shape (number_of_query_set_images, **image_shape)

        Returns:
            torch.Tensor: shape(n_query*n_way, n_way), classification prediction for each query data
        """
        pass

    def query_process(self, query_images):
        return query_images

    def fit_on_task(
        self, support_images, support_labels, query_images, query_labels, optimizer
    ):
        """
        Perform a forward pass and a backward pass on one episode.
        Args:
            support_images (torch.Tensor): shape (number_of_support_set_images, **image_shape)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
            query_images (torch.Tensor): shape (number_of_query_set_images, **image_shape)
            query_labels (torch.Tensor): artificial query set labels in range (0, n_way)
            optimizer (torch.optim.Optimizer): model optimizer

        Returns:
            tuple(torch.Tensor, torch.Tensor): detached from the computational graph
                - shape(n_query*n_way, n_way), classification prediction for each query data
                - shape(,), training loss
        """
        optimizer.zero_grad()
        scores = self.set_forward(support_images, support_labels, query_images)
        loss = self.loss_fn(scores, query_labels)
        loss.backward()
        optimizer.step()

        return scores.detach(), loss.detach().item()

    def extract_features(self, support_images, query_images):
        """
        Computes the features vectors of the support and query sets
        Args:
            support_images (torch.Tensor): shape (n_support_images, **image_dim) input data
            query_images (torch.Tensor): shape (n_query_images, **image_dim) input data

        Returns:
            Tuple(torch.Tensor, torch.Tensor): features vectors of the support and query sets, respectively of shapes
            (n_support_images, features_dim) and (n_query_images, features_dim)
        """
        # Set to CUDA if available
        support_images = set_device(support_images)
        query_images = set_device(query_images)

        z_support = self.feature.forward(support_images)
        z_query = self.feature.forward(query_images)

        return z_support, z_query

    @staticmethod
    def get_prototypes(features, labels):
        """
        Compute a prototype for each class.
        Args:
            features (torch.Tensor): shape[n_images, feature_dim], feature vectors
            labels (torch.Tensor): shape[n_images], label corresponding to each feature vector
        Returns:
            torch.Tensor: prototypes. shape[number_of_unique_values_in_labels, feature_dim]
        """

        n_way = len(torch.unique(labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        return torch.cat(
            [features[torch.nonzero(labels == label)].mean(0) for label in range(n_way)]
        )

    @staticmethod
    def evaluate(scores, query_labels):
        """
        Predict labels of query images and returns the number of correct top1 predictions
        Args:
            scores (torch.Tensor): shape (number_of_query_set_images, n_way)
            query_labels (torch.Tensor): artificial query set labels in range (0, n_way)

        Returns:
            float: classification accuracy in [0,1]
        """

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        top1_correct = float((topk_labels[:, 0] == query_labels).sum())
        return float(top1_correct) / len(query_labels)

    @staticmethod
    def get_task_perf(
        task_id, classification_scores, labels, class_ids, source_id, target_id
    ):
        """
        Records the classification results for each query instance.
        Args:
            task_id (int): index of the task
            classification_scores (torch.Tensor): predicted classification scores
            labels (torch.Tensor): ground truth labels
            class_ids (list[int]): indices of the classes composing the current classification task
            source_id (int): index of the source domain for this task
            target_id (int): index of the target domain for this task

        Returns:
            pd.DataFrame: for each couple (query, class), gives classification score, class_id,
                ground truth query label, current task id and source and target domain,
        """
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "task_id": task_id,
                        "source_domain": source_id,
                        "target_domain": target_id,
                        "image_id": i,
                        "true_label": class_ids[labels[i]],
                        "predicted_label": [
                            class_ids[label]
                            for label in range(classification_scores.shape[1])
                        ],
                        "score": classification_scores[i],
                    }
                )
                for i in range(labels.shape[0])
            ]
        )

    def train_loop(self, epoch, train_loader, optimizer):
        """
        Executes one training epoch
        Args:
            epoch (int): current epoch
            train_loader (DataLoader): loader of a given number of episodes
            optimizer (torch.optim.Optimizer): model optimizer

        Returns:
            tuple(float, float): resp. average loss and classification accuracy
        """
        print_freq = 100

        loss_list = []
        acc_list = []

        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
            _,
            _,
        ) in enumerate(train_loader):
            if (
                support_images.shape[0] == 1
            ):  # the batch size is 1 while using meta-dataset
                support_images = support_images.squeeze(0)
                support_labels = support_labels.squeeze(0)
                query_images = query_images.squeeze(0)
                query_labels = query_labels.squeeze(0)
            query_labels = set_device(query_labels)
            scores, loss_value = self.fit_on_task(
                support_images, support_labels, query_images, query_labels, optimizer
            )

            loss_list.append(loss_value)

            acc_list.append(self.evaluate(scores, query_labels) * 100)

            if episode_index % print_freq == print_freq - 1:
                logger.info(
                    "Epoch {epoch} | Batch {episode_index}/{n_batches} | Loss {loss}".format(
                        epoch=epoch,
                        episode_index=episode_index + 1,
                        n_batches=len(train_loader),
                        loss=np.asarray(loss_list).mean(),
                    )
                )
            # torch.cuda.empty_cache()

        return np.asarray(loss_list).mean(), np.asarray(acc_list).mean()

    def eval_loop(self, test_loader):  # META_DATASET_VERSION
        """

        Args:
            test_loader (DataLoader): loader of a given number of episodes

        Returns:
            tuple(float, float, pd.DataFrame): resp. average loss and classification accuracy,
                and advanced evaluation statistics
        """
        acc_all = []
        loss_all = []
        evaluation_stats = []
        n_tasks = len(test_loader)
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
            source_domain,
            target_domain,
        ) in tqdm(enumerate(test_loader)):
            if (
                support_images.shape[0] == 1
            ):  # the batch size is 1 while using meta-dataset
                support_images = support_images.squeeze(0)
                support_labels = support_labels.squeeze(0)
                query_images = query_images.squeeze(0)
                query_labels = query_labels.squeeze(0)
            query_labels = set_device(query_labels)
            query_images = set_device(query_images)
            query_images = self.query_process(query_images)
            support_images = self.query_process(set_device(support_images))
            # with torch.no_grad():
            scores = self.set_forward(
                support_images, support_labels, query_images
            ).detach()
            # scores = self.set_forward(support_images, support_labels, query_images).detach()

            loss_value = self.loss_fn(scores, query_labels).detach().item()

            loss_all.append(loss_value)
            acc_all.append(
                self.evaluate(scores.detach().cpu(), query_labels.detach().cpu()) * 100
            )
            del support_images, support_labels, query_images, query_labels
            # torch.cuda.empty_cache()

        # evaluations_stats_df = pd.concat(evaluation_stats, ignore_index=True)
        evaluations_stats_df = None
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        logger.info(
            "%d Test Accuracy = %4.2f%% +- %4.2f%%"
            % (n_tasks, acc_mean, confidence_interval(acc_std, n_tasks))
        )

        return (
            np.asarray(loss_all).mean(),
            (acc_mean, confidence_interval(acc_std, n_tasks)),
            evaluations_stats_df,
        )


from configs import experiment_config


class AbsMetaLearnerR(AbstractMetaLearner):
    def _do_nothing(self, input):
        return input

    def query_process(self, query_images):
        if INVERSE_RESIZE:
            query_images = self.inverse_resize(query_images)
        return query_images

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_resize = self._do_nothing
