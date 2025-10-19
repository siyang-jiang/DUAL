from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from configs import dataset_config
from configs import experiment_config
from src.data_tools.utils import episodic_collate_fn
from configs.erm_training_config import N_WORKERS


class LossList:
    def __init__(self):
        self.last_value = 0.0
        self.sum_value = 0.0
        self.length = 0

    def append(self, value):
        self.last_value = (
            value.detach().item() if type(value) == torch.Tensor else value
        )
        self.sum_value = self.sum_value + self.last_value
        self.length += 1

    def __len__(self):
        return self.length

    def mean(self):
        if self.length == 0:
            return 0.0
        return self.sum_value / self.length

    def last(self):
        return self.last_value


class LossManager:
    def __init__(self, loss_names):
        self.losses = {names: LossList() for names in loss_names}

    def reset(self):
        self.losses = {names: LossList() for names in self.losses.keys()}

    def single_reset(self, loss_name):
        self.losses[loss_name] = LossList()

    def batch_append(self, values: dict):
        for names, value in values.items():
            self.losses[names].append(value)

    def single_append(self, key, value):
        self.losses[key].append(value)

    def get_dicts(self, loss_list=None, type="mean", with_sum=False):
        loss_list = self.losses.keys() if loss_list is None else loss_list
        outputs = {key: getattr(self.losses[key], type)() for key in loss_list}
        if with_sum:
            outputs["Sum"] = self.get_losses_sum(loss_list, type)
        return outputs

    def get_outputs(self, loss_list=None, type="mean", with_sum=False):
        loss_list = self.losses.keys() if loss_list is None else loss_list
        outputs = ""
        for key in loss_list:
            value = getattr(self.losses[key], type)()
            outputs += f"{key}:{value:.4f}, "
        if with_sum:
            outputs += f"Sum:{self.get_losses_sum(loss_list, type):.4f}, "
        return outputs[:-2]

    def get_losses_sum(self, loss_list=None, type="mean"):
        loss_list = self.losses.keys() if loss_list is None else loss_list
        sum_value = 0.0
        for key in loss_list:
            if key != "acc":
                sum_value += getattr(self.losses[key], type)()
        return sum_value

    def __call__(self, loss_name):
        return self.losses[loss_name]


def set_device(x):
    """
    Switch a tensor to GPU if CUDA is available, to CPU otherwise
    """
    device = f"cuda:{experiment_config.GPU_ID}" if torch.cuda.is_available() else "cpu"
    return x.to(device=device)


def plot_episode(support_images, query_images):
    """
    Plot images of an episode, separating support and query images.
    Args:
        support_images (torch.Tensor): tensor of multiple-channel support images
        query_images (torch.Tensor): tensor of multiple-channel query images
    """

    def matplotlib_imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    support_grid = torchvision.utils.make_grid(support_images)
    matplotlib_imshow(support_grid)
    plt.title("support images")
    plt.show()
    query_grid = torchvision.utils.make_grid(query_images)
    plt.title("query images")
    matplotlib_imshow(query_grid)
    plt.show()


def elucidate_ids(df, dataset):
    """
    Retrieves explicit class and domain names in dataset from their integer index,
        and returns modified DataFrame
    Args:
        df (pd.DataFrame): input DataFrame. Must be the same format as the output of AbstractMetaLearner.get_task_perf()
        dataset (Dataset): the dataset
    Returns:
        pd.DataFrame: output DataFrame with explicit class and domain names
    """
    return df.replace(
        {
            "predicted_label": dataset.id_to_class,
            "true_label": dataset.id_to_class,
            "source_domain": dataset.id_to_domain,
            "target_domain": dataset.id_to_domain,
        }
    )


def get_episodic_loader(
    split: str, n_way: int, n_source: int, n_target: int, n_episodes: int
):
    dataset = dataset_config.DATASET(
        dataset_config.DATA_ROOT, split, dataset_config.IMAGE_SIZE
    )
    sampler = dataset.get_sampler()(
        n_way=n_way,
        n_source=n_source,
        n_target=n_target,
        n_episodes=n_episodes,
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=N_WORKERS,
            pin_memory=False,  # SY changed
            collate_fn=episodic_collate_fn,
        ),
        dataset,
    )
