import os
import torch
import argparse
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from src.config import MODEL_CONFIG, TEST_DIR, TRAIN_CONFIG, OUTPUT_DIR
from src.models.builder import build_models  # 用于构建模型架构
from src.training.model_trainer.base import BaseGANTrainer

MODEL_CONFIG["model_name"] = "stylegan"  # 确保使用对应模型架构
#CKPT = os.path.join(OUTPUT_DIR, "dcgan_full","epoch_090_interval.pth")  # 默认检查点路径
#CKPT = os.path.join(OUTPUT_DIR, "lsgan_full","epoch_175_interval.pth")  # 默认检查点路径
#CKPT = os.path.join(OUTPUT_DIR, "wgangp_full","epoch_175_interval.pth")  # 默认检查点路径
CKPT = os.path.join(OUTPUT_DIR, "stylegan_full","epoch_045_interval.pth")  # 默认检查点路径

def load_checkpoint(G, D, ckpt_path, device):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    G.load_state_dict(ckpt["G_state_dict"])
    D.load_state_dict(ckpt["D_state_dict"])

    return G, D


def generate_images(G, num, z_dim, save_path, nrow=4):
    G.eval()
    with torch.no_grad():
        z = torch.randn(num, z_dim, device=next(G.parameters()).device)
        fake = G(z)
        fake = (fake + 1) / 2  # [-1,1] → [0,1]

        grid = vutils.make_grid(fake, nrow=nrow, normalize=False, padding=2)
        ndarr = grid.permute(1, 2, 0).cpu().numpy()

        # 保存图片
        plt.figure(figsize=(nrow * 2, nrow * 2))
        plt.imshow(ndarr)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f"[INFO] Generated images saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CKPT, help="Path to checkpoint .pth file")
    parser.add_argument("--num", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--out", type=str, default="test_samples.jpg", help="Output image file name")
    parser.add_argument("--nrow", type=int, default=4, help="Images per row")

    args = parser.parse_args()

    device = TRAIN_CONFIG["device"]
    z_dim = MODEL_CONFIG["z_dim"]

    # -----------------------
    # 1) 构建模型架构
    # -----------------------
    G, D = build_models()
    G = G.to(device)
    D = D.to(device)

    # -----------------------
    # 2) 加载权重
    # -----------------------
    G, D = load_checkpoint(G, D, args.ckpt, device)

    # -----------------------
    # 3) 生成图片
    # -----------------------
    TEST_OUT_DIR = os.path.join(TEST_DIR, MODEL_CONFIG["model_name"])
    os.makedirs(TEST_OUT_DIR, exist_ok=True)
    save_path = os.path.join(TEST_OUT_DIR, args.out)

    generate_images(G, args.num, z_dim, save_path, args.nrow)


if __name__ == "__main__":
    main()
