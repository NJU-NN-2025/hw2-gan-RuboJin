# HW2-炉石传说Full-art 图像生成

简介
----
本项目目标是使用多种生成对抗网络（GAN）——包括 LSGAN、DCGAN、WGAN-GP 与 StyleGAN——对《炉石传说》卡牌的 Full-art 原画进行建模与生成。
数据集来源于暴雪官方`https://blizzard.gamespress.com`，
并经过统一预处理与增强（将原始 1091 张扩增至 2182 张）后用于训练。

项目结构（重要文件/目录）
-------------------------
```bibtex
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── models/
│   ├── training/
│   ├── data/
│   ├── utils/
│   ├── config.py
│   ├── test.py
│   └── main.py
│
├── scripts/
├── test/
├── train.sh
├── test.sh
├── requirements.txt
└── output/

```
- data/raw/                原始抓取到的 Full-art 原画（1024×1024 原图，未裁剪或仅做最少处理）
- data/processed/          训练用预处理后数据（例如 128、64 等分辨率）
- data/processed/examples/ 预处理前后的样例对比图
- scripts/fetch.py         抓取 Full-art 图片的脚本（可指定最大数量、自动去重、过滤 icon 等）
- scripts/preprocess.py    预处理脚本（裁剪 / 缩放 / 增强 *2 扩充）
- scripts/trans.py         一些辅助的图像转换工具
- src/config.py            全局与模型配置（分辨率、batch size、模型选择等）
- src/data/dataset.py      数据集 & DataLoader 构建接口
- src/models/              模型实现目录（每个 GAN 的 generator/discriminator）
- src/models/builder.py    从 src/models 下查找并构建指定模型
- src/training/model_trainer/  每个 GAN 的训练器（如 lsgan_t.py, dcgan_t.py, stylegan_t.py）
- src/training/model_trainer/base.py  统一的训练框架（日志、保存、度量）
- src/main.py              训练入口（加载配置、数据、模型并启动训练）
- src/test.py              用于生成指定模型样本并保存至 test/ 下
- output/                  训练输出（checkpoints、samples、log）
- requirements.txt         依赖项列表
- train.sh / test.sh       Linux 下的便捷运行脚本（Windows 下直接调用 python 即可）

## 快速开始

### 训练示例

在默认配置下（要切换 GAN 类型、学习率、scheduler 等，请修改：
`src/config.py` 中的 `MODEL_CONFIG` 或 `TRAIN_CONFIG`。
）：

Linux:
```bash
bash train.sh --model dcgan
```

Windows (PowerShell):
```powershell
python src/main.py --model dcgan
```

常见参数（在 `src/config.py` 中调整）：
- model: dcgan | lsgan | wgangp | stylegan
- img_size: 输入图像尺寸（例如 64、128、256）
- batch_size: 每批样本数（参考下文推荐）
- epochs: 训练轮数
- save_every_epoch: 是否每轮保存检查点
- save_best_enable / save_criterion: 是否启用保存最优模型及评判指标
- lr_scheduler: 动态学习率调度器类型

### 如何选取batch_size

- 64×64 分辨率下，RTX 4060 最推荐 batch_size = 64（稳定且速度快）
- StyleGAN 结构较重，建议 batch_size = 16 或 8
- 若显存不足则优先降低 batch_size 而不是降低 image_size

### 训练保存策略（最佳模型与最后模型）

- 保存策略(可于config内修改保存策略)：
  1. 每 N (可配置)轮保存一次检查点。
  2. 同时维护一个 `last.pth`（每轮覆盖）用于快速恢复最近训练状态。
  3. 使用一个评估指标（g+d,可配置），当指标达到新最好时保存为 `best.pth`（并保留最好的若干个以便回溯）。

- 实现细节：
  - `save_best_enable` 启用后，仅在检测到评分真正优于当前 best 时才保存一次 `best.pth`（而不是每个 epoch 都保存）。
  - 若想只保留最终的 best 而非每次都保存，则在保存新 best 后可以删除旧 best 的文件名或覆盖同一文件名。


### 训练技巧总结

- DCGAN/LSGAN：调小 learning rate（如 2e-4），使用 Adam (beta1=0.5, beta2=0.999)，batch size 尝试 32/64。使用谱归一化或批归一化提高稳定性。
- WGAN-GP：使用 RMSProp 或 Adam（beta1=0.0, beta2=0.9），使用梯度惩罚（gp）并增加 critic 更新次数（n_critic = 5），不要在 discriminator 使用批归一化。
- StyleGAN：结构更复杂（映射网络、样式注入、逐层噪声），注意渐进训练或小学习率、更复杂的正则化与权重初始化,注意要求时间或算力充足。
- 实际训练中，LSGAN 和 DCGAN 最容易在高 epoch (>150) 后出现模式坍缩，因此建议配合学习率衰减。

### 常见问题（FAQ）

Q: 训练很慢，tqdm 长时间不动？  
A: 首次迭代或数据第一次传到 GPU 会较为缓慢。

### 不足与改进之处
- 超参数调优过程较为困难,因此可能模型没有展现出最佳水平
- 部分模型（如 StyleGAN）实现较为简化，未包含所有原论文细节,且由于超参数问题并无良好结果
- 各个GAN的训练流程可以优化以增强效率

### 许可与引用

Full-art 图像的版权归原作者/Blizzard所有，本项目仅用于研究与学习目的。

### AI声明

>部分代码结构规划与文档格式由 ChatGPT 辅助生成,本人已对所有代码与流程进行理解、修改与调试.

欢迎提出 Issue 或 PR，与我讨论与改进本项目。
