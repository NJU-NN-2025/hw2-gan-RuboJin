import random
import argparse
import numpy as np
import torch

from src.config import RANDOM_SEED, MODEL_CONFIG, TRAIN_CONFIG
from src.models.builder import build_models
from src.training.trainer import build_trainer
from src.data.dataset import build_dataloader


# ------------------------------------------------------------
# 固定随机种子
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# 创建命令行参数
# ------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train GAN Models")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--z_dim", type=int, default=None)
    parser.add_argument("--lr_G", type=float, default=None)
    parser.add_argument("--lr_D", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)

    return parser


# ------------------------------------------------------------
# 覆盖 config 里的默认参数
# ------------------------------------------------------------
def override_config(args):
    # MODEL_CONFIG
    if args.model_name is not None:
        MODEL_CONFIG["model_name"] = args.model_name

    if args.image_size is not None:
        MODEL_CONFIG["image_size"] = args.image_size

    if args.z_dim is not None:
        MODEL_CONFIG["z_dim"] = args.z_dim

    # TRAIN_CONFIG
    if args.epochs is not None:
        TRAIN_CONFIG["epochs"] = args.epochs

    if args.batch_size is not None:
        TRAIN_CONFIG["batch_size"] = args.batch_size

    if args.lr_G is not None:
        TRAIN_CONFIG["lr_G"] = args.lr_G

    if args.lr_D is not None:
        TRAIN_CONFIG["lr_D"] = args.lr_D


    print("\n[INFO] ===== Effective CONFIG =====")
    print("MODEL_CONFIG:", MODEL_CONFIG)
    print("TRAIN_CONFIG:", {k: v for k, v in TRAIN_CONFIG.items() if k not in ["lr_scheduler"]})
    print("=================================\n")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    # 加载命令行参数
    parser = build_arg_parser()
    args = parser.parse_args()

    # 覆盖配置
    override_config(args)

    # 固定随机种子
    set_seed(RANDOM_SEED)

    # 创建 dataloader / model / trainer
    dataloader = build_dataloader()
    G, D = build_models()

    trainer = build_trainer(
        gan_type=MODEL_CONFIG["model_name"],
        G=G, D=D, dataloader=dataloader
    )

    trainer.train()


if __name__ == "__main__":
    main()
