import os

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.preprocess import preprocess


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    preprocess()
    print(f"[OK] Preprocessing done. Output dir: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
