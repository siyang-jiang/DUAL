import argparse
import os
from distutils.dir_util import copy_tree


def cover(l, var_list):
    for i, var in enumerate(var_list):
        if f"{var} = " in l:
            return i
    return -1


def rewrite_test(EXP_CONFIG, LRS):
    ## exp config
    with open(EXP_CONFIG, "r") as read:
        lines = read.readlines()
    with open(EXP_CONFIG, "w") as w:
        for l in lines:
            if "SMART_RESIZER =" in l:
                w.write(f"SMART_RESIZER = {LRS}\n")
            else:
                w.write(l)


def rewrite(args):
    ## exp config
    var_list = [
        "PGADA",
        "GPU_ID",
        "MODEL",
        "N_SHOT",
        "N_TARGET",
        "BATCH",
        "EXP_NAME",
        "PROPORTION",
        "MULTI_PERTUBATION",
        "FIX_G",
        "ENCODING_IN_PHI",
        "FORCE_OT",
    ]
    val_list = [
        (not args.noPGADA),
        args.gpu,
        args.model,
        args.shot,
        args.target,
        f'"{args.batch}"',
        f'"{args.exp_name}"',
        args.p_rate,
        args.multi_p,
        args.fix_g,
        args.encoding_r,
        args.testing_ot,
    ]
    if args.model == 4:
        args.tp = True
    else:
        args.tp = False
    with open(args.exp_config, "r") as read:
        lines = read.readlines()
    with open(args.exp_config, "w") as w:
        for l in lines:
            if "=" in l:
                tag = l.split(" = ")[0]
                if tag in var_list:
                    i = var_list.index(l.split(" = ")[0])
                    output = f"{var_list[i]} = {val_list[i]}\n"
                    w.write(output)
                    print(output[:-1])
                    continue
            w.write(l)
    read.close()
    if args.model == 4:
        args.model = 0

    ## model config
    with open(args.model_config, "r") as read:
        lines = read.readlines()
    tp = ", transportation=TRANSPORTATION_MODULE" if args.tp else ""
    with open(args.model_config, "w") as w:
        flag = -1
        for l in lines:
            # MODEL = partial(model_list[0], transportation=TRANSPORTATION_MODULE)
            if "MODEL = " in l:
                w.write(f"MODEL = partial(model_list[{args.model}]{tp})")
            else:
                w.write(l)
    read.close()

    ## exp config
    with open(args.training_config, "r") as read:
        lines = read.readlines()
    with open(args.training_config, "w") as w:
        for l in lines:
            if "N_SOURCE = " in l:
                w.write(f"N_SOURCE = {args.shot}\n")
                continue
            w.write(l)
    read.close()
    print("Finish Config Experiments")


def dataset_rewrite(args):
    dataset_list = [
        "mini_imagenet_c_config",
        "cifar_100_c_config",
        "femnist_config",
        "regular_tiered_imagenet_c_config",
        "meta_dataset_config",
    ]
    ## dataset config
    with open(args.dataset_config, "w") as w:
        w.write(
            f"from configs.all_datasets_configs.{dataset_list[args.dataset]} import *"
        )

    ## Model config
    with open(args.model_config, "r") as read:
        lines = read.readlines()
    with open(args.model_config, "w") as w:
        flag = -1
        model = (
            "ResNet18"
            if args.dataset == 0 or args.dataset == 3 or args.dataset == 4
            else "Conv4"
        )
        r_model = "R_4_ADV" if args.dataset == 2 else "R_RED"
        for l in lines:
            if "BACKBONE = " in l:
                w.write(f"BACKBONE = {model}\n")
                continue
            if "R = " in l:
                w.write(f"R = {r_model}\n")
                continue
            w.write(l)

    target = 1 if args.dataset == 2 else args.target
    ## exp config
    with open(args.training_config, "r") as read:
        lines = read.readlines()
    with open(args.training_config, "w") as w:
        for l in lines:
            if "N_TARGET = " in l:
                w.write(f"N_TARGET = {target}\n")
                continue
            w.write(l)

    print("Finish Config Dataset Experiments")
    ## erm training config
    with open(args.erm_training_config, "r") as read:
        lines = read.readlines()
    with open(args.erm_training_config, "w") as w:
        for l in lines:
            if "SIMCLR = " in l:
                w.write(f"SIMCLR = {bool(args.SIMCLR)}\n")
                continue
            w.write(l)


def main():
    parser = argparse.ArgumentParser(description="Process rewrite configs")
    parser.add_argument("--testing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--testing_lrs", action=argparse.BooleanOptionalAction)
    parser.add_argument("--noPGADA", action=argparse.BooleanOptionalAction)

    parser.add_argument("--batch", type=str, default="TEST", help="batch name")
    parser.add_argument("--exp_name", type=str, default="", help="")

    parser.add_argument("--SIMCLR", type=int, default=1, help="SIMCLR")
    parser.add_argument("--p_rate", type=float, default=0.5, help="proportion")
    parser.add_argument("--multi_p", type=int, default=-1, help="-1 for random apply")
    parser.add_argument("--fix_g", type=int, default=0, help="fix g")
    parser.add_argument("--encoding_r", type=int, default=0, help="encoding in r")
    parser.add_argument("--testing_ot", type=int, default=1, help="OT_testing")

    parser.add_argument("--dataset", type=int, default=1, help="Change the dataset")
    parser.add_argument("--quest", type=int, default=0, help="Change the quest")
    parser.add_argument("--level", type=int, default=0, help="Change the quest level")
    parser.add_argument("--gpu", type=int, default=0, help="GPU")

    parser.add_argument("--model", type=int, default=0, help="model number")
    parser.add_argument("--shot", type=int, default=1, help="N_SHOT")
    parser.add_argument("--target", type=int, default=1, help="N_target")

    parser.add_argument(
        "--exp_config",
        type=str,
        default="./configs/experiment_config.py",
        help="experiment config path",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/model_config.py",
        help="model config path",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="./configs/training_config.py",
        help="training config path",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="./configs/dataset_config.py",
        help="dataset config path",
    )
    parser.add_argument(
        "--erm_training_config",
        type=str,
        default="./configs/erm_training_config.py",
        help="erm training config path",
    )
    dataset_list = [
        "mini_imagenet",
        "cifar100",
        "femnist",
        "tiered_imagenet",
        "meta_dataset",
    ]

    args = parser.parse_args()
    if args.exp_name == "":
        args.exp_name = dataset_list[args.dataset]
    copy_tree("configs_template", "configs")
    rewrite(args=args)
    dataset_rewrite(args)
    if args.testing or args.testing_lrs:
        log_name = "smart_testing" if args.testing_lrs else "testing_trival"
        if not args.testing_ot:
            log_name += "_without_OT"
        rewrite_test(args.exp_config, args.testing_lrs)
        with_str = "with" if args.testing_lrs else "without"
        print(f"Config loaded, start testing {with_str} repairer...")
        if args.dataset == 4:
            os.system(f"python3 -m scripts.test_meta --name {log_name}")
        else:
            os.system(f"python3 -m scripts.erm_testing --name {log_name}")
        return
    print("Config loaded, start training...")
    if args.noPGADA:
        rewrite_test(args.exp_config, False)
        main_path = (
            "scripts.train_meta_baseline"
            if args.dataset == 4
            else "scripts.run_experiment"
        )
        os.system(f"python3 -m {main_path}")
    else:
        os.system("python3 -m scripts.erm_training")


if __name__ == "__main__":
    main()
