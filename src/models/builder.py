from src.config import MODEL_CONFIG, BUILDER_CONFIG
from src.models.architectures.dcgan import DCGANGenerator, DCGANDiscriminator
from src.models.architectures.wgan import WGANGenerator, WGANCritic
from src.models.architectures.lsgan import LSGANGenerator, LSGANDiscriminator
from src.models.architectures.stylegan import StyleGAN2Generator, StyleGAN2Discriminator

def build_models():
    model_name = MODEL_CONFIG["model_name"]
    G, D = None, None
    img_size = MODEL_CONFIG["image_size"]
    if model_name == "dcgan":
        G = DCGANGenerator(
            z_dim=MODEL_CONFIG["z_dim"],
            img_size=img_size,
            img_channels=MODEL_CONFIG["img_channels"],
            feature_maps=MODEL_CONFIG["feature_maps"]
        )
        D = DCGANDiscriminator(
            img_channels=MODEL_CONFIG["img_channels"],
            img_size=img_size,
            feature_maps=MODEL_CONFIG["feature_maps"]
        )
    elif model_name == "wgangp":
        G = WGANGenerator(
            z_dim=MODEL_CONFIG["z_dim"],
            img_channels=MODEL_CONFIG["img_channels"],
            img_size=img_size,
            feature_maps=MODEL_CONFIG["feature_maps"]
        )
        D = WGANCritic(
            img_channels=MODEL_CONFIG["img_channels"],
            img_size=img_size,
            feature_maps=MODEL_CONFIG["feature_maps"]
        )
    elif model_name == "lsgan":
        G = LSGANGenerator(
            z_dim=MODEL_CONFIG["z_dim"],
            img_channels=MODEL_CONFIG["img_channels"],
            img_size=img_size,
            feature_maps=MODEL_CONFIG["feature_maps"]
        )
        D = LSGANDiscriminator(
            img_channels=MODEL_CONFIG["img_channels"],
            img_size=img_size,
            feature_maps=MODEL_CONFIG["feature_maps"]
        )
    elif model_name == "stylegan":
        # 未来扩展
        G = StyleGAN2Generator(
            z_dim=MODEL_CONFIG["z_dim"],
            img_channels=MODEL_CONFIG["img_channels"],
            img_size=img_size,
            fmap=MODEL_CONFIG["feature_maps"]
        )
        D = StyleGAN2Discriminator(
            img_channels=MODEL_CONFIG["img_channels"],
            img_size=img_size,
            fmap=MODEL_CONFIG["feature_maps"]
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 初始化方式处理
    init_type = BUILDER_CONFIG.get("init_type", "normal")
    # 这里可以做初始化函数调用
    if init_type == "normal":
        pass
    return G, D
