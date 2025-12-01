import torch
import torch.nn as nn

from src.training.model_trainer.base import BaseGANTrainer
# -----------------------------
# LSGAN Trainer (MSE loss)
# -----------------------------
class LSGANTrainer(BaseGANTrainer):
    def __init__(self, G, D, dataloader):
        super().__init__(G, D, dataloader)
        self.criterion = nn.MSELoss()

    def train_step(self, real):
        batch = real.size(0)
        real = real.to(self.device)

        real_labels = torch.ones(batch, device=self.device)
        fake_labels = torch.zeros(batch, device=self.device)

        # --- train D ---
        z = torch.randn(batch, self.z_dim, device=self.device)
        fake = self.G(z).detach()
        d_real = self.D(real)
        d_fake = self.D(fake)

        d_loss = 0.5 * (self.criterion(d_real, real_labels) + self.criterion(d_fake, fake_labels))
        self.opt_D.zero_grad()
        d_loss.backward()
        self.opt_D.step()

        # --- train G ---
        z = torch.randn(batch, self.z_dim, device=self.device)
        fake = self.G(z)
        d_fake = self.D(fake)
        g_loss = 0.5 * self.criterion(d_fake, real_labels)
        self.opt_G.zero_grad()
        g_loss.backward()
        self.opt_G.step()

        return d_loss.item(), g_loss.item()

