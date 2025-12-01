import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from src.config import PROCESSED_DATA_DIR, DATALOADER_CONFIG


class CardArtDataset(Dataset):
    def __init__(self, data_dir=PROCESSED_DATA_DIR, image_size=64, augment=False, normalize=True):
        self.paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}
        ]
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize

        # 变换定义
        transform_list = []
        if augment:
            transform_list += [
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
            ]
        transform_list.append(T.Resize((image_size, image_size)))
        transform_list.append(T.ToTensor())

        if normalize:
            # GAN 默认输入范围 [-1, 1]
            transform_list.append(T.Normalize(mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5]))

        self.transform = T.Compose(transform_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, 0  # 第二个返回值是标签占位符


def build_dataloader():
    """根据 config 构建 dataloader"""
    batch_size = DATALOADER_CONFIG["batch_size"]
    num_workers = DATALOADER_CONFIG["num_workers"]
    shuffle = DATALOADER_CONFIG["shuffle"]
    augment = DATALOADER_CONFIG.get("augment", False)
    normalize = DATALOADER_CONFIG.get("normalize", True)

    dataset = CardArtDataset(augment=augment, normalize=normalize)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True
    )
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")
    return dataloader
