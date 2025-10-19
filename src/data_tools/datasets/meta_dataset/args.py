import argparse
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser("Few-shot learning script", add_help=False)
    # General
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--name", type=str, default="testing_trival", help="logfile name"
    )
    parser.add_argument("--debug", type=int, default=0, help="debug flag")
    parser.add_argument("--save_dir", type=str, default="", help="logfile name")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.set_defaults(fp16=True)
    parser.add_argument(
        "--output_dir",
        default="outputs/tmp",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--data_path", default="data/meta_dataset", help="path to dataset"
    )
    parser.add_argument(
        "--device", default="cuda", help="cuda:gpu_id for single GPU training"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--nEpisode", default=400, type=int)
    # MetaDataset parameters
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Images will be resized to this value",
    )
    parser.add_argument(
        "--base_sources",
        nargs="+",
        default=[
            "aircraft",
            "cu_birds",
            "dtd",
            "fungi",
            "ilsvrc_2012",
            "omniglot",
            "quickdraw",
            "vgg_flower",
        ],
        help="List of datasets to use for training",
    )
    parser.add_argument(
        "--val_sources",
        nargs="+",
        default=[
            "aircraft",
            "cu_birds",
            "dtd",
            "fungi",
            "ilsvrc_2012",
            "omniglot",
            "quickdraw",
            "vgg_flower",
        ],
        help="List of datasets to use for validation",
    )
    parser.add_argument(
        "--test_sources",
        nargs="+",
        default=[
            "traffic_sign",
            "mscoco",
            "ilsvrc_2012",
            "omniglot",
            "aircraft",
            "cu_birds",
            "dtd",
            "quickdraw",
            "fungi",
            "vgg_flower",
        ],
        help="List of datasets to use for meta-testing",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether or not to shuffle data for TFRecordDataset",
    )
    parser.add_argument(
        "--train_transforms",
        nargs="+",
        default=[
            "random_resized_crop",
            "jitter",
            "random_flip",
            "to_tensor",
            "normalize",
        ],
        help="Transforms applied to training data",
    )
    parser.add_argument(
        "--test_transforms",
        nargs="+",
        default=["resize", "center_crop", "to_tensor", "normalize"],
        help="Transforms applied to test data",
    )
    parser.add_argument(
        "--num_ways",
        type=int,
        default=None,
        help="Set it if you want a fixed # of ways per task",
    )
    parser.add_argument(
        "--num_support",
        type=int,
        default=None,
        help="Set it if you want a fixed # of support samples per class",
    )
    parser.add_argument(
        "--num_query",
        type=int,
        default=None,
        help="Set it if you want a fixed # of query samples per class",
    )
    parser.add_argument(
        "--min_ways", type=int, default=5, help="Minimum # of ways per task"
    )
    parser.add_argument(
        "--max_ways_upper_bound",
        type=int,
        default=20,
        help="Maximum # of ways per task",
    )  # 50 originally
    parser.add_argument(
        "--max_num_query", type=int, default=10, help="Maximum # of query samples"
    )
    parser.add_argument(
        "--max_support_set_size",
        type=int,
        default=200,
        help="Maximum # of support samples",
    )  # 500 originally
    parser.add_argument(
        "--max_support_size_contrib_per_class",
        type=int,
        default=100,
        help="Maximum # of support samples per class",
    )
    parser.add_argument(
        "--min_examples_in_class",
        type=int,
        default=0,
        help="Classes that have less samples will be skipped",
    )
    parser.add_argument(
        "--min_log_weight",
        type=float,
        default=np.log(0.5),
        help="Do not touch, used to randomly sample support set",
    )
    parser.add_argument(
        "--max_log_weight",
        type=float,
        default=np.log(2),
        help="Do not touch, used to randomly sample support set",
    )
    parser.add_argument(
        "--ignore_bilevel_ontology",
        action="store_true",
        help="Whether or not to use superclass for BiLevel datasets (e.g Omniglot)",
    )
    parser.add_argument(
        "--ignore_dag_ontology",
        action="store_true",
        help="Whether to ignore ImageNet DAG ontology when sampling \
                              classes from it. This has no effect if ImageNet is not  \
                              part of the benchmark.",
    )
    parser.add_argument(
        "--ignore_hierarchy_probability",
        type=float,
        default=0.0,
        help="if using a hierarchy, this flag makes the sampler \
                              ignore the hierarchy for this proportion of episodes \
                              and instead sample categories uniformly.",
    )

    return parser
