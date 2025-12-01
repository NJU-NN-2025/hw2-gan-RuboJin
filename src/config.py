import os
import torch
from src.utils.io_tools import ensure_dir

PROJECT_ROOT= os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
TEST_DIR = os.path.join(PROJECT_ROOT, "test")

ensure_dir(RAW_DATA_DIR)
ensure_dir(OUTPUT_DIR)
ensure_dir(TEST_DIR)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 69
EPOCH = 45  #与下文一致,方便更改
LR_G = 0.001
LR_D = 0.001 #WGAN不使用外置LR控制,请进入TRAIN_CONFIG修改
BATCH = 64
VIR = 15 #可视化间隔
CHECK = 15 #检查点保存间隔




PREPROCESS_CONFIG = {
    "image_size": 64, # 预处理后图像大小
    "aug_factor": 2, # 每张图像增强次数（含原图）
    "example_num": 5, # 最多展示多少张增强样例
}

MODEL_CONFIG = {
    "model_name": "stylegan", # 模型名称，可选 "dcgan", "wgangp", "lsgan", "stylegan"
    "z_dim": 128, # 潜在向量维度
    "img_channels": 3, # 图像通道数
    "feature_maps": 128, # 特征图数量基数
    "image_size":PREPROCESS_CONFIG["image_size"],
}

DATALOADER_CONFIG = {
    "batch_size": BATCH, # 批量大小
    "num_workers": os.cpu_count()//2, # 数据加载线程数
    "shuffle": True, # 是否打乱数据
    "augment": False, # 是否进行数据增强
    "normalize": True, # 是否归一化图像
}

BUILDER_CONFIG = {
    "use_ema": False, # 是否使用指数移动平均
    "init_type": "normal", # 初始化方式
}

TRAIN_CONFIG = {
    "device": DEVICE,
    "batch_size": DATALOADER_CONFIG["batch_size"], # 批量大小
    "lr_G": LR_G, # 学习率
    "lr_D": LR_D, # 学习率
    "betas": (0.0, 0.99), # Adam优化器:(0.5, 0.999)通常用于GAN,(0.0,0.99)用于StyleGAN
    "real_label_smooth": 0.9, # 真实标签平滑
    "fake_label_smooth": 0.0, # 伪造标签平滑
    "wgan_lr" : 1e-4, # WGAN的学习率
    "wgan_betas": (0.0, 0.9), # WGAN的Adam优化器参数
    "lambda_gp": 1.0, # WGAN-GP的梯度惩罚系数
    "n_critic": 1, # WGAN-GP的判别器更新次数,若为styleGAN则推荐为1

    "r1_gamma": 5.0, # R1正则化强度
    "r1_interval": 16, # 每多少步应用一次R1正则化
    "mixing_prob": 0.9, # 风格混合概率

    "epochs": EPOCH, # 训练轮数
    "visualize_interval": VIR, # 可视化间隔
    "mini_sample":False, # 是否使用了小样本进行快速测试
    "checkpoint_interval": CHECK, # 检查点保存间隔

    "save_best": False, # 是否保存最佳模型
    "save_last": True, # 是否保存最后模型
    "save_best_at_end": False, # 是否在训练结束时保存最佳模型
    "save_every_epoch": False, # 是否每轮保存模型
    "save_criterion":"g+d", # 保存最佳模型的标准，可选 "g_loss", "d_loss","g+d"

    "lr_scheduler": {
        "enable": True,
        #"type": "step",
        #"type": "exp",
        #"type": "multistep",
        "type": "cosine",  # 余弦退火
        "milestones_G": [25, 60,120,160],  # G的衰减点
        "gamma_G": 0.90,  # G的衰减比例 (较慢)
        "milestones_D": [25, 60,120,160],  # D的衰减点
        "gamma_D": 0.5,  # D的衰减比例 (较快/剧烈)

        "step_size": 25,  # 衰减步长
        "gamma": 0.7,  # 衰减比例
    }

}

