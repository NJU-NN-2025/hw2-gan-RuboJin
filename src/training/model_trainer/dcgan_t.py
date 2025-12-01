import torch
import torch.nn as nn

from src.training.model_trainer.base import BaseGANTrainer
from src.config import TRAIN_CONFIG

# -----------------------------
# DCGAN Trainer
# -----------------------------
class DCGANTrainer(BaseGANTrainer):
    def __init__(self, G, D, dataloader):
        super().__init__(G, D, dataloader)
        self.criterion = nn.BCEWithLogitsLoss()

        # Label smoothing parameters
        self.real_label = TRAIN_CONFIG.get("real_label_smooth", 0.9)
        self.fake_label = TRAIN_CONFIG.get("fake_label_smooth", 0.1)

    def train_step(self, real):
        batch = real.size(0)
        real = real.to(self.device)

        # Label smoothing
        real_labels = torch.full((batch,), self.real_label, device=self.device)
        fake_labels = torch.full((batch,), self.fake_label, device=self.device)

        # --- train D ---

        z = torch.randn(batch, self.z_dim, device=self.device)
        fake = self.G(z).detach()

        d_real = self.D(real)
        d_fake = self.D(fake)

        d_loss = self.criterion(d_real, real_labels) + self.criterion(d_fake, fake_labels)
        self.opt_D.zero_grad()
        d_loss.backward()
        self.opt_D.step()

        # --- train G ---
        z = torch.randn(batch, self.z_dim, device=self.device)
        fake = self.G(z)
        d_fake = self.D(fake)

        # G 用 real_label_smooth (0.9)，效果最佳
        g_labels = torch.full((batch,), self.real_label, device=self.device)
        g_loss = self.criterion(d_fake, g_labels)

        self.opt_G.zero_grad()
        g_loss.backward()
        self.opt_G.step()

        return d_loss.item(), g_loss.item()
