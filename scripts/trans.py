import os
from PIL import Image
from tqdm import tqdm

from src.config import RAW_DATA_DIR

INPUT_DIR = "D:/360MoveData/Users/lenovo/Desktop/Hearthstone"        # 原画根目录
OUTPUT_DIR = RAW_DATA_DIR           # 输出目录
TARGET_SIZE = 1024                  # 输出尺寸 1024x1024
OUTPUT_EXT = "png"                  # "png" 或 "jpg"


def is_image_file(name):
    ext = os.path.splitext(name)[1].lower()
    return ext in {".jpg", ".jpeg", ".png"}


def load_all_images(root):
    """递归加载所有图片路径"""
    paths = []
    for base, dirs, files in os.walk(root):
        for f in files:
            if is_image_file(f):
                paths.append(os.path.join(base, f))
    return paths


def convert_and_save(paths):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, path in enumerate(tqdm(paths, desc="Processing")):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

            new_name = f"card_{idx:05d}.{OUTPUT_EXT}"
            save_path = os.path.join(OUTPUT_DIR, new_name)

            if OUTPUT_EXT == "jpg":
                img.save(save_path, quality=95)
            else:
                img.save(save_path)

        except Exception as e:
            print(f"[WARN] Failed: {path}: {e}")


def main():
    print(f"[INFO] Scanning images in {INPUT_DIR} ...")
    paths = load_all_images(INPUT_DIR)
    print(f"[INFO] Found {len(paths)} images.")

    convert_and_save(paths)

    print(f"[DONE] Converted images saved in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
