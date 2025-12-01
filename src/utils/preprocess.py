import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PREPROCESS_CONFIG

OUT_DIR = PROCESSED_DATA_DIR
EXAMPLE_DIR = os.path.join(OUT_DIR, "examples")

TARGET_SIZE = PREPROCESS_CONFIG["image_size"]      # 输出分辨率
AUG_FACTOR = PREPROCESS_CONFIG["aug_factor"]          # 每张增强次数（含原图 = 1 + 增强次数）
MAX_EXAMPLES = PREPROCESS_CONFIG["example_num"]      # 最多展示多少张增强样例
EXTS = {".png", ".jpg", ".jpeg"}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_image(path):
    return Image.open(path).convert("RGB")


# -----------------------
# 轻增强
# -----------------------
def light_augment(img):
    # 随机轻裁剪
    if random.random() < 0.6:
        w, h = img.size
        crop_ratio = random.uniform(0.92, 0.98)
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        x1 = random.randint(0, w - new_w)
        y1 = random.randint(0, h - new_h)
        img = img.crop((x1, y1, x1 + new_w, y1 + new_h))

    # 小角度旋转
    if random.random() < 0.3:
        angle = random.uniform(-1.5, 1.5)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    # light color jitter
    if random.random() < 0.7:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.95, 1.05))
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.97, 1.03))
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.97, 1.03))

    return img


def resize_to_square(img, size):
    return img.resize((size, size), Image.LANCZOS)


# -----------------------
# 拼接图像（左：原图，右：处理后）
# -----------------------
def concat_side_by_side(left, right):
    w1, h1 = left.size
    w2, h2 = right.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), (0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w1, 0))
    return canvas


# -----------------------
# 主流程
# -----------------------
def preprocess():
    ensure_dir(OUT_DIR)
    ensure_dir(EXAMPLE_DIR)

    paths = [
        os.path.join(RAW_DATA_DIR, f)
        for f in os.listdir(RAW_DATA_DIR)
        if os.path.splitext(f)[1].lower() in EXTS
    ]

    print(f"[INFO] Found {len(paths)} raw images.")

    idx = 0
    example_count=0
    for p in tqdm(paths, desc="Processing"):
        img = load_image(p)

        for k in range(AUG_FACTOR):
            if k == 0:
                proc = img.copy()
            else:
                proc = light_augment(img.copy())

            proc = resize_to_square(proc, TARGET_SIZE)

            # 保存训练用图
            out_path = os.path.join(OUT_DIR, f"card_{idx:05d}.jpg")
            proc.save(out_path, quality=95, optimize=True)
            idx += 1

            # 保存预览 examples（最多 5 个）
            if example_count < MAX_EXAMPLES:
                preview = concat_side_by_side(
                    img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS),
                    proc
                )
                example_path = os.path.join(
                    EXAMPLE_DIR,
                    f"example_{os.path.splitext(os.path.basename(p))[0]}.jpg"
                )
                preview.save(example_path)
                example_count += 1

    print(f"[DONE] Generated {idx} processed images")
    print(f"[DONE] Example previews saved in {EXAMPLE_DIR}")


if __name__ == "__main__":
    preprocess()
