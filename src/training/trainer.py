from src.config import TRAIN_CONFIG
from src.training.model_trainer.dcgan_t import DCGANTrainer
from src.training.model_trainer.lsgan_t import LSGANTrainer
from src.training.model_trainer.wgan_t import WGANGPTrainer
from src.training.model_trainer.stylegan_t import StyleGAN2Trainer


def build_trainer(gan_type, G, D, dataloader):

    gan_type = gan_type.lower()

    if gan_type == "stylegan":
        return StyleGAN2Trainer(
            G, D, dataloader,
            r1_gamma=TRAIN_CONFIG["r1_gamma"],
            r1_interval=TRAIN_CONFIG["r1_interval"]
        )

    mapping = {
        "dcgan": DCGANTrainer,
        "lsgan": LSGANTrainer,
        "wgangp": WGANGPTrainer,
    }

    cls = mapping.get(gan_type)
    if cls is None:
        raise ValueError(f"Unknown gan_type {gan_type}")

    return cls(G, D, dataloader)
