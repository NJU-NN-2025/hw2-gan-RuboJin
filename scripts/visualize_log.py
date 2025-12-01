import pandas as pd
import matplotlib.pyplot as plt
import os
from src.config import OUTPUT_DIR, MODEL_CONFIG, TRAIN_CONFIG


def visualize_log(csv_path=None):
    """训练结束后自动绘制 D/G loss 曲线"""
    import pandas as pd
    csv_path = csv_path
    if not os.path.exists(csv_path):
        print("[WARN] No CSV log found, skip visualization.")
        print("Usage: python scripts/visualize_log.py <path_to_csv>")
        return

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["d_loss"], label="D Loss")
    plt.plot(df["epoch"], df["g_loss"], label="G Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{MODEL_CONFIG['model_name'].upper()} Training Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[VISUALIZE] Saved loss curve to {save_path}")


if __name__ == "__main__":
    visualize_log()
