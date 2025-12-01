import torch
import torch.optim as optim

from src.config import TRAIN_CONFIG
from src.training.model_trainer.base import BaseGANTrainer

# -----------------------------
# WGAN-GP Trainer
# -----------------------------
class WGANGPTrainer(BaseGANTrainer):
    def __init__(self, G, D, dataloader):
        super().__init__(G, D, dataloader)
        # optimizer params for WGAN-GP
        lr = TRAIN_CONFIG.get("wgan_lr", 1e-4)
        betas = TRAIN_CONFIG.get("wgan_betas", (0.0, 0.9))
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=betas)

        self.lambda_gp = TRAIN_CONFIG.get("lambda_gp", 10.0)
        self.n_critic = TRAIN_CONFIG.get("n_critic", 5)

    def compute_gp(self, real, fake):
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = alpha * real + (1 - alpha) * fake
        interpolates.requires_grad_(True)

        d_interpolates = self.D(interpolates)
        grad_outputs = torch.ones_like(d_interpolates, device=self.device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def train_step(self, real):
        batch = real.size(0)
        real = real.to(self.device)

        # train critic n_critic times
        d_loss_total = 0.0
        for _ in range(self.n_critic):
            z = torch.randn(batch, self.z_dim, device=self.device)
            fake = self.G(z).detach()

            d_real = self.D(real).mean()
            d_fake = self.D(fake).mean()
            gp = self.compute_gp(real, fake)

            d_loss = -(d_real - d_fake) + self.lambda_gp * gp

            self.opt_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            d_loss_total += d_loss.item()

        # train generator
        z = torch.randn(batch, self.z_dim, device=self.device)
        fake = self.G(z)
        g_loss = -self.D(fake).mean()

        self.opt_G.zero_grad()
        g_loss.backward()
        self.opt_G.step()

        return d_loss_total / self.n_critic, g_loss.item()
