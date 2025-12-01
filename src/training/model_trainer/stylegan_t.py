import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.model_trainer.base import BaseGANTrainer


class StyleGAN2Trainer(BaseGANTrainer):
    def __init__(self, G, D, dataloader, r1_gamma=10.0, r1_interval=16):
        super().__init__(G, D, dataloader)
        self.r1_gamma = r1_gamma
        self.r1_interval = r1_interval

    # logistic loss for discriminator
    def d_logistic_loss(self, real_pred, fake_pred):
        return (F.softplus(-real_pred) + F.softplus(fake_pred)).mean()

    # non-saturating loss for generator
    def g_nonsat_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def compute_r1_penalty(self, real_img):
        real_img = real_img.requires_grad_(True)
        real_pred = self.D(real_img)

        grad = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_img,
            create_graph=True
        )[0]

        penalty = grad.pow(2).reshape(real_img.size(0), -1).sum(1).mean()
        return penalty

    def train_step(self, real):
        batch = real.size(0)
        real = real.to(self.device)


        # ==================================
        # 1) train D
        # ==================================
        z = torch.randn(batch, self.z_dim, device=self.device)
        fake = self.G(z).detach()

        real_pred = self.D(real)
        fake_pred = self.D(fake)

        d_loss = self.d_logistic_loss(real_pred, fake_pred)

        # R1 regularization
        if self.global_step % self.r1_interval == 0:
            r1_penalty = self.compute_r1_penalty(real)
            d_loss = d_loss + self.r1_gamma * r1_penalty * 0.5

        self.opt_D.zero_grad()
        d_loss.backward()

        self.opt_D.step()

        # ==================================
        # 2) train G
        # ==================================
        z = torch.randn(batch, self.z_dim, device=self.device)
        fake = self.G(z)
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=5.0)
        fake_pred = self.D(fake)

        g_loss = self.g_nonsat_loss(fake_pred)

        self.opt_G.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=5.0)
        self.opt_G.step()

        # update global step
        self.global_step += 1

        return d_loss.item(), g_loss.item()
