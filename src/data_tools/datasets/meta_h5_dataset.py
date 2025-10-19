import os
import random
import h5py
from PIL import Image
import json

# import cv2
import numpy as np

import torch
from src.data_tools.datasets.meta_dataset import config as config_lib
from src.data_tools.datasets.meta_dataset import sampling
from src.data_tools.datasets.meta_dataset.utils import Split
from src.data_tools.datasets.meta_dataset.transform import get_transforms
from src.data_tools.datasets.meta_dataset import dataset_spec as dataset_spec_lib
from src.data_tools.utils import get_perturbations

from torchvision import transforms
import torchvision.transforms.functional as TF
from configs.dataset_specs.tiered_imagenet_c.perturbation_params import (
    PERTURBATION_PARAMS,
)
from src.data_tools.transform import TransformLoader


class perturbator:
    def __init__(self, image_size, split_spec_path) -> None:
        self.image_size = image_size
        with open(split_spec_path) as file:
            self.split_specs = json.load(file)
        self.perturbations, self.id_to_domain = get_perturbations(
            self.split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )
        from configs.experiment_config import PROPORTION  # avoid circular import
        from configs.experiment_config import MULTI_PERTUBATION

        self.multi_p_type = int(MULTI_PERTUBATION)
        self.p_rate = PROPORTION if MULTI_PERTUBATION == -1 else 1.0

    def _get_perturbation_list(self):
        perturbation_list = []
        pertubation_types = (
            random.sample(self.perturbations.keys(), self.multi_p_type)
            if self.multi_p_type > 0
            else self.perturbations.keys()
        )
        for types in pertubation_types:
            if random.random() < self.p_rate:  # Hyperparameter
                perturbation_list.append(random.choice(self.perturbations[types]))
        return perturbation_list

    def torture(self, image, perturbations=None):
        r"""
        Args:
            image (numpy or tensor): input image
            perturbations (list): list of perturbations

        Returns:
            pertubated image (tensor, float)
        """
        assert type(image) == np.ndarray or type(image) == torch.Tensor, (
            f"Input image type is {type(image)}, not numpy or tensor."
        )
        if perturbations is None:
            perturbations = self._get_perturbation_list()
        if image.shape[2] > 3:
            image = image.numpy()
            image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
        assert (
            image.shape[0] == self.image_size
            and image.shape[1] == self.image_size
            and image.shape[2] == 3
        ), f"Input image shape is {image.shape}"

        for p in perturbations:
            image = p(image).astype(np.uint8)
        return TF.to_tensor(image)


class FullMetaDatasetH5(torch.utils.data.Dataset):
    def __init__(self, args, split=Split["TRAIN"]):
        super().__init__()

        # Data & episodic configurations
        data_config = config_lib.DataConfig(args)
        episod_config = config_lib.EpisodeDescriptionConfig(args)
        self.toTensor = transforms.ToTensor()
        self.PILtoTensor = transforms.PILToTensor()
        if split == Split.TRAIN:
            datasets = args.base_sources
            episod_config.num_episodes = args.nEpisode
        elif split == Split.VALID:
            datasets = args.val_sources
            episod_config.num_episodes = args.nValEpisode
        else:
            datasets = args.test_sources
            episod_config.num_episodes = 1000

        self.image_size = args.image_size

        use_dag_ontology_list = [False] * len(datasets)
        use_bilevel_ontology_list = [False] * len(datasets)
        if episod_config.num_ways:
            if len(datasets) > 1:
                raise ValueError("For fixed episodes, not tested yet on > 1 dataset")
        else:
            # Enable ontology aware sampling for Omniglot and ImageNet.
            if "omniglot" in datasets:
                use_bilevel_ontology_list[datasets.index("omniglot")] = True
            if "ilsvrc_2012" in datasets:
                use_dag_ontology_list[datasets.index("ilsvrc_2012")] = True

        episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
        episod_config.use_dag_ontology_list = use_dag_ontology_list

        # dataset specifications
        all_dataset_specs = []
        for dataset_name in datasets:
            dataset_records_path = os.path.join(data_config.path, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            all_dataset_specs.append(dataset_spec)

        num_classes = sum(
            [len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs]
        )
        print(
            f"=> There are {num_classes} classes in the {split} split of the combined datasets"
        )
        self.num_classes = num_classes

        self.datasets = datasets
        self.transforms = get_transforms(data_config, split)
        self.len = episod_config.num_episodes * len(
            datasets
        )  # NOTE: not all datasets get equal number of episodes per epoch

        self.class_map = {}  # 2-level dict of h5 paths
        self.class_h5_dict = {}  # 2-level dict of opened h5 files
        self.class_samplers = {}  # 1-level dict of samplers, one for each dataset
        self.class_images = {}  # 2-level dict of image ids, one list for each class

        for i, dataset_name in enumerate(datasets):
            dataset_spec = all_dataset_specs[i]
            base_path = dataset_spec.path
            class_set = dataset_spec.get_classes(split)  # class ids in this split
            num_classes = len(class_set)

            record_file_pattern = dataset_spec.file_pattern
            assert record_file_pattern.startswith("{}"), (
                f"Unsupported {record_file_pattern}."
            )

            self.class_map[dataset_name] = {}
            self.class_h5_dict[dataset_name] = {}
            self.class_images[dataset_name] = {}

            for class_id in class_set:
                data_path = os.path.join(
                    base_path, record_file_pattern.format(class_id)
                )
                self.class_map[dataset_name][class_id] = data_path.replace(
                    "tfrecords", "h5"
                )
                self.class_h5_dict[dataset_name][class_id] = None  # closed h5 is None
                self.class_images[dataset_name][class_id] = [
                    str(j)
                    for j in range(dataset_spec.get_total_images_per_class(class_id))
                ]

            self.class_samplers[dataset_name] = sampling.EpisodeDescriptionSampler(
                dataset_spec=dataset_spec,
                split=split,
                episode_descr_config=episod_config,
                use_dag_hierarchy=episod_config.use_dag_ontology_list[i],
                use_bilevel_hierarchy=episod_config.use_bilevel_ontology_list[i],
                ignore_hierarchy_probability=args.ignore_hierarchy_probability,
            )
            self.perturbator = perturbator(
                self.image_size,
                f"configs/dataset_specs/meta_dataset/{split.name.lower()}.json",
            )

    def __len__(self):
        return self.len

    def _tensorize(self, image_batch):
        return [
            self.toTensor(img) if type(img) != torch.Tensor else img
            for img in image_batch
        ]

    def get_next(self, source, class_id, idx):
        # fetch h5 path
        h5_path = self.class_map[source][class_id]

        # load h5 file if None
        if (
            self.class_h5_dict[source][class_id] is None
        ):  # will be closed in the end of main.py
            self.class_h5_dict[source][class_id] = h5py.File(h5_path, "r")

        h5_file = self.class_h5_dict[source][class_id]
        record = h5_file[idx]
        x = record["image"][()]
        return x

    def __getitem__(self, idx):
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        # select which dataset to form episode
        source = np.random.choice(self.datasets)
        sampler = self.class_samplers[source]

        # episode details: (class_id, nb_supp, nb_qry)
        episode_description = sampler.sample_episode_description()
        episode_description = tuple(  # relative ids --> abs ids
            (class_id + sampler.class_set[0], num_support, num_query)
            for class_id, num_support, num_query in episode_description
        )
        episode_classes = list({class_ for class_, _, _ in episode_description})

        for class_id, nb_support, nb_query in episode_description:
            assert nb_support + nb_query <= len(self.class_images[source][class_id]), (
                f"Failed fetching {nb_support + nb_query} images from {source} at class {class_id}."
            )
            random.shuffle(self.class_images[source][class_id])

            # support
            for j in range(0, nb_support):
                # print('support fetch:', sup_added, class_id)
                x = self.get_next(
                    source, class_id, self.class_images[source][class_id][j]
                )
                if x.shape[-1] == 1:
                    x = np.repeat(x, 3, axis=-1)
                if x.shape[1] != self.image_size or x.shape[2] != self.image_size:
                    x = (
                        TF.resize(TF.to_tensor(x), (self.image_size, self.image_size))
                        .numpy()
                        .transpose(1, 2, 0)
                    )
                support_images.append(TF.to_tensor(x))

            # query
            for j in range(nb_support, nb_support + nb_query):
                x = self.get_next(
                    source, class_id, self.class_images[source][class_id][j]
                )
                if x.shape[-1] == 1:
                    x = np.repeat(x, 3, axis=-1)
                if x.shape[1] != self.image_size or x.shape[2] != self.image_size:
                    x = (
                        TF.resize(TF.to_tensor(x), (self.image_size, self.image_size))
                        .numpy()
                        .transpose(1, 2, 0)
                    )
                query_images.append(TF.to_tensor(x))

            support_labels.extend([episode_classes.index(class_id)] * nb_support)
            query_labels.extend([episode_classes.index(class_id)] * nb_query)

        support_images = torch.stack(support_images, dim=0)
        query_images = torch.stack(query_images, dim=0)

        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        if support_images.shape[2] != self.image_size:
            support_images = TF.resize(
                support_images, (self.image_size, self.image_size)
            )
            query_images = TF.resize(query_images, (self.image_size, self.image_size))

        for i, img in enumerate(support_images):
            support_images[i] = self.perturbator.torture(img)
        for i, img in enumerate(query_images):
            query_images[i] = self.perturbator.torture(img)

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_id,
            0,
            0,
        )


class FullMetaDatasetH5_ERM(FullMetaDatasetH5):
    def __init__(self, args, split=Split["TRAIN"], SIMCLR=False):
        if SIMCLR:
            super(FullMetaDatasetH5_ERM, self).__init__(args, Split["TEST"])
        else:
            super(FullMetaDatasetH5_ERM, self).__init__(
                args, Split["TRAIN"] if split != Split.TEST else split
            )
        self.SIMCLR = SIMCLR
        self.SIMCLR_val = SIMCLR and split == Split.VALID

        if split != Split.TEST or SIMCLR:  # divide the dataset into train and val
            for source in self.class_images.keys():
                for class_id in self.class_images[source]:
                    border = int(len(self.class_images[source][class_id]) * 0.8)
                    self.class_images[source][class_id] = (
                        self.class_images[source][class_id][:border]
                        if split == Split.TRAIN
                        else self.class_images[source][class_id][border:]
                    )
        self.split = split
        # self.len = 1000000
        if SIMCLR:
            self.transform = TransformLoader(self.image_size).get_composed_transform(
                aug=True
            )
            self.transform_test = TransformLoader(
                self.image_size
            ).get_composed_transform(aug=False)

    def __len__(self):
        return 9999999

    def __getitem__(self, idx):
        # select which dataset to form episode
        source = np.random.choice(self.datasets)
        sampler = self.class_samplers[source]

        # episode details: (class_id, nb_supp, nb_qry)
        episode_description = sampler.sample_episode_description()
        if not self.SIMCLR:
            episode_description = tuple(  # relative ids --> abs ids
                (class_id + sampler.class_set[0], num_support, num_query)
                for class_id, num_support, num_query in episode_description
            )
        episode_classes = list({class_ for class_, _, _ in episode_description})

        class_id = episode_description[0][0] + sampler.class_set[0]
        image_id = random.randint(
            0, len(self.class_images[source][class_id]) - 1
        )  # random pick one image
        img = self.get_next(
            source, class_id, self.class_images[source][class_id][image_id]
        )
        if img.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        if img.shape[1] != self.image_size or img.shape[2] != self.image_size:
            img = TF.resize(TF.to_tensor(img), (self.image_size, self.image_size))
        label = class_id
        img_p = self.perturbator.torture(img)

        if self.SIMCLR:
            img = TF.to_pil_image(img)
            img1 = (
                self.transform_test(img) if self.SIMCLR_val else self.transform(img)
            )  # Should the SIMCLR learn from the perturbated image?
            img2 = self.transform(img)
            image_batch = self._tensorize(
                [img1, img_p, img, img_p]
            )  # [img1, img2, pure_img, pure_img_p]
        else:
            image_batch = self._tensorize([img, img_p])  # [pure_img, pure_img_p]
        return image_batch, label, 0  # perturbation_index


class FullMetaDatasetH5_old(torch.utils.data.Dataset):
    def __init__(self, args, split=Split["TRAIN"]):
        super().__init__()

        # Data & episodic configurations
        data_config = config_lib.DataConfig(args)
        episod_config = config_lib.EpisodeDescriptionConfig(args)

        if split == Split.TRAIN:
            datasets = args.base_sources
            episod_config.num_episodes = args.nEpisode
        elif split == Split.VALID:
            datasets = args.val_sources
            episod_config.num_episodes = args.nValEpisode
        else:
            datasets = args.test_sources
            episod_config.num_episodes = 600

        use_dag_ontology_list = [False] * len(datasets)
        use_bilevel_ontology_list = [False] * len(datasets)
        if episod_config.num_ways:
            if len(datasets) > 1:
                raise ValueError("For fixed episodes, not tested yet on > 1 dataset")
        else:
            # Enable ontology aware sampling for Omniglot and ImageNet.
            if "omniglot" in datasets:
                use_bilevel_ontology_list[datasets.index("omniglot")] = True
            if "ilsvrc_2012" in datasets:
                use_dag_ontology_list[datasets.index("ilsvrc_2012")] = True

        episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
        episod_config.use_dag_ontology_list = use_dag_ontology_list

        # dataset specifications
        all_dataset_specs = []
        for dataset_name in datasets:
            dataset_records_path = os.path.join(data_config.path, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            all_dataset_specs.append(dataset_spec)

        num_classes = sum(
            [len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs]
        )
        print(
            f"=> There are {num_classes} classes in the {split} split of the combined datasets"
        )

        self.datasets = datasets
        self.transforms = get_transforms(data_config, split)
        self.len = episod_config.num_episodes * len(
            datasets
        )  # NOTE: not all datasets get equal number of episodes per epoch

        self.class_map = {}  # 2-level dict of h5 paths
        self.class_h5_dict = {}  # 2-level dict of opened h5 files
        self.class_samplers = {}  # 1-level dict of samplers, one for each dataset
        self.class_images = {}  # 2-level dict of image ids, one list for each class

        for i, dataset_name in enumerate(datasets):
            dataset_spec = all_dataset_specs[i]
            base_path = dataset_spec.path
            class_set = dataset_spec.get_classes(split)  # class ids in this split
            num_classes = len(class_set)

            record_file_pattern = dataset_spec.file_pattern
            assert record_file_pattern.startswith("{}"), (
                f"Unsupported {record_file_pattern}."
            )

            self.class_map[dataset_name] = {}
            self.class_h5_dict[dataset_name] = {}
            self.class_images[dataset_name] = {}

            for class_id in class_set:
                data_path = os.path.join(
                    base_path, record_file_pattern.format(class_id)
                )
                self.class_map[dataset_name][class_id] = data_path.replace(
                    "tfrecords", "h5"
                )
                self.class_h5_dict[dataset_name][class_id] = None  # closed h5 is None
                self.class_images[dataset_name][class_id] = [
                    str(j)
                    for j in range(dataset_spec.get_total_images_per_class(class_id))
                ]

            self.class_samplers[dataset_name] = sampling.EpisodeDescriptionSampler(
                dataset_spec=dataset_spec,
                split=split,
                episode_descr_config=episod_config,
                use_dag_hierarchy=episod_config.use_dag_ontology_list[i],
                use_bilevel_hierarchy=episod_config.use_bilevel_ontology_list[i],
                ignore_hierarchy_probability=args.ignore_hierarchy_probability,
            )

    def __len__(self):
        return self.len

    def get_next(self, source, class_id, idx):
        # fetch h5 path
        h5_path = self.class_map[source][class_id]

        # load h5 file if None
        if (
            self.class_h5_dict[source][class_id] is None
        ):  # will be closed in the end of main.py
            self.class_h5_dict[source][class_id] = h5py.File(h5_path, "r")

        h5_file = self.class_h5_dict[source][class_id]
        record = h5_file[idx]
        x = record["image"][()]

        if self.transforms:
            x = Image.fromarray(x)
            x = self.transforms(x)

        return x

    def __getitem__(self, idx):
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        # select which dataset to form episode
        source = np.random.choice(self.datasets)
        sampler = self.class_samplers[source]

        # episode details: (class_id, nb_supp, nb_qry)
        episode_description = sampler.sample_episode_description()
        episode_description = tuple(  # relative ids --> abs ids
            (class_id + sampler.class_set[0], num_support, num_query)
            for class_id, num_support, num_query in episode_description
        )
        episode_classes = list({class_ for class_, _, _ in episode_description})

        for class_id, nb_support, nb_query in episode_description:
            assert nb_support + nb_query <= len(self.class_images[source][class_id]), (
                f"Failed fetching {nb_support + nb_query} images from {source} at class {class_id}."
            )
            random.shuffle(self.class_images[source][class_id])

            # support
            for j in range(0, nb_support):
                # print('support fetch:', sup_added, class_id)
                x = self.get_next(
                    source, class_id, self.class_images[source][class_id][j]
                )
                support_images.append(x)

            # query
            for j in range(nb_support, nb_support + nb_query):
                x = self.get_next(
                    source, class_id, self.class_images[source][class_id][j]
                )
                query_images.append(x)

            support_labels.extend([episode_classes.index(class_id)] * nb_support)
            query_labels.extend([episode_classes.index(class_id)] * nb_query)

        support_images = torch.stack(support_images, dim=0)
        query_images = torch.stack(query_images, dim=0)

        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels


if __name__ == "__main__":  # test
    import warnings

    warnings.filterwarnings("ignore")
    from src.data_tools.datasets.meta_dataset.args import get_args_parser
    from src.data_tools.datasets.meta_dataset import config as config_lib
    from src.data_tools.datasets.meta_dataset.utils import Split
    from src.data_tools.datasets.meta_h5_dataset import FullMetaDatasetH5
    from src.data_tools.datasets.meta_val_dataset import MetaValDataset

    warnings.filterwarnings("ignore")

    args = get_args_parser().parse_args()
    args.base_sources = ["ilsvrc_2012"]
    args.nValEpisode = 120
    args.image_size = 128
    args.min_ways = 5
    args.max_ways_upper_bound = 20
    args.num_support = 5
    args.num_query = 5
    # trainSet = FullMetaDatasetH5_ERM(args, Split.TRAIN, SIMCLR=True)
    # valSet = FullMetaDatasetH5_ERM(args, Split.TEST)
    # sample = [trainSet.__getitem__(0) for i in range(100)]

    valSet = MetaValDataset(
        os.path.join(
            args.data_path,
            "ilsvrc_2012",
            f"val_ep{args.nValEpisode}_img{args.image_size}.h5",
        ),
        num_episodes=args.nValEpisode,
    )
    val_loader = torch.utils.data.DataLoader(
        valSet, batch_size=1, num_workers=os.cpu_count()
    )
    sample = next(iter(val_loader))
    print("test done")
