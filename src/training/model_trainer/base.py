import math
import os
import csv
import shutil
import torch
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR,
    CosineAnnealingLR, ExponentialLR
)

from src.config import TRAIN_CONFIG, MODEL_CONFIG, OUTPUT_DIR


# ================================
# Base Trainer (Generic GAN logic)
# ================================
class BaseGANTrainer:
    def __init__(self, G, D, dataloader):
        self.device = TRAIN_CONFIG["device"]
        self.G = G.to(self.device)
        self.D = D.to(self.device)
        self.dataloader = dataloader

        # ---- Optimizers ----
        lr_G = TRAIN_CONFIG.get("lr_G")
        lr_D = TRAIN_CONFIG.get("lr_D")
        betas = TRAIN_CONFIG.get("betas", (0.5, 0.999))

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_G, betas=betas)
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_D, betas=betas)

        # ---- z noise ----
        self.z_dim = MODEL_CONFIG["z_dim"]
        self.fixed_z = torch.randn(16, self.z_dim, device=self.device)

        # ---- output directory ----
        self.out_dir = os.path.join(
            OUTPUT_DIR,
            f"{MODEL_CONFIG['model_name']}_{'mini' if TRAIN_CONFIG.get('mini_sample', False) else 'full'}"
        )
        self._prepare_output_dir()

        # ---- optional dynamic LR ----
        self._build_lr_scheduler()

        # ---- training state (for resuming) ----
        self.global_step = 0

        # ---- best-model saving config ----
        # TRAIN_CONFIG may include:
        #   save_best: bool (enable saving the best model, default True)
        #   save_last: bool (save final/last checkpoint, default True)
        #   save_criterion: str one of ('g_loss','d_loss','g+d') to decide best model (default 'g_loss')
        #   save_every_epoch: bool (save checkpoint each epoch in addition to best/last, default False)
        #   save_best_at_end: bool (if True, keep best weights in memory and write a single best file at end)
        self.save_best_enable = TRAIN_CONFIG.get('save_best', True)
        self.save_last = TRAIN_CONFIG.get('save_last', True)
        self.save_every_epoch = TRAIN_CONFIG.get('save_every_epoch', False)
        self.save_criterion = TRAIN_CONFIG.get('save_criterion', 'g_loss')
        self.save_best_at_end = TRAIN_CONFIG.get('save_best_at_end', False)
        # whether we minimize the score (losses) or maximize (e.g., metrics like FID inverted)
        self._minimize = self.save_criterion in ('g_loss', 'd_loss', 'g+d')
        self.best_score = None
        self.best_epoch = None

        # when save_best_at_end=True we keep a snapshot of the best states in-memory
        self._best_state = None

    # ======================================================
    # LR Scheduler creation
    # ======================================================
    def _build_lr_scheduler(self):
        sched_cfg = TRAIN_CONFIG.get("lr_scheduler", {})
        if not sched_cfg.get("enable", False):
            self.scheduler_G = None
            self.scheduler_D = None
            print("[INFO] LR scheduler disabled.")
            return

        typ = sched_cfg.get("type", "step")

        if typ == "step":
            self.scheduler_G = StepLR(self.opt_G, sched_cfg["step_size"], sched_cfg["gamma"])
            self.scheduler_D = StepLR(self.opt_D, sched_cfg["step_size"], sched_cfg["gamma"])

        elif typ == "multistep":
            milestones_G = sched_cfg.get("milestones_G", sched_cfg.get("milestones"))
            gamma_G = sched_cfg.get("gamma_G", sched_cfg.get("gamma", 0.1))
            self.scheduler_G = MultiStepLR(self.opt_G, milestones_G, gamma_G)
            # 独立配置 D 的参数
            milestones_D = sched_cfg.get("milestones_D", sched_cfg.get("milestones"))
            gamma_D = sched_cfg.get("gamma_D", sched_cfg.get("gamma", 0.1))
            self.scheduler_D = MultiStepLR(self.opt_D, milestones_D, gamma_D)

        elif typ == "cosine":
            T_max = sched_cfg.get("T_max", TRAIN_CONFIG["epochs"])
            self.scheduler_G = CosineAnnealingLR(self.opt_G, T_max)
            self.scheduler_D = CosineAnnealingLR(self.opt_D, T_max)

        elif typ == "exp":
            self.scheduler_G = ExponentialLR(self.opt_G, sched_cfg["gamma"])
            self.scheduler_D = ExponentialLR(self.opt_D, sched_cfg["gamma"])

        else:
            raise ValueError(f"Unknown scheduler type: {typ}")

        print(f"[INFO] LR scheduler enabled: {typ}")

    # ======================================================
    # Score computation & checkpointing helpers
    # ======================================================
    def _compute_score(self, d_loss, g_loss):
        """
        Compute a single scalar score from d/g losses according to self.save_criterion.
        Lower is better when self._minimize is True.
        """
        crit = (self.save_criterion or 'g_loss').lower()
        if crit == 'g_loss':
            return float(g_loss)
        elif crit == 'd_loss':
            return float(d_loss)
        elif crit == 'g+d' or crit == 'd+g':
            return float(g_loss + d_loss)
        else:
            # default fallback to generator loss
            return float(g_loss)

    def _checkpoint_path(self, epoch, tag):
        # e.g. best_g_loss_epoch_010.pth or last_epoch_050.pth
        crit = self.save_criterion.replace(' ', '')
        if tag == 'best':
            name = f"best_{crit}_epoch_{epoch:03d}.pth"
        elif tag == 'last':
            name = f"last_epoch_{epoch:03d}.pth"
        else:
            name = f"epoch_{epoch:03d}_{tag}.pth"
        return os.path.join(self.out_dir, name)

    def _save_checkpoint(self, epoch, tag='last'):
        path = self._checkpoint_path(epoch, tag)
        state = {
            'epoch': epoch,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'train_config': TRAIN_CONFIG,
            'model_config': MODEL_CONFIG,
        }
        torch.save(state, path)
        print(f"[CHECKPOINT] Saved {tag} checkpoint: {path}")

    def _maybe_save_best_and_last(self, epoch, d_mean, g_mean):
        # Save every-epoch if requested
        if self.save_every_epoch:
            self._save_checkpoint(epoch, tag=f'epoch')

        # Save interval checkpoint
        if epoch % TRAIN_CONFIG.get("checkpoint_interval", 25) == 0:
            self._save_checkpoint(epoch, tag=f'interval')

        # Save last
        if self.save_last and epoch == TRAIN_CONFIG["epochs"]:
            self._save_checkpoint(epoch, tag='last')

        # Best model logic
        if not self.save_best_enable:
            return

        score = self._compute_score(d_mean, g_mean)
        # initialize best
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch

            if self.save_best_at_end:
                # just keep the state in memory and defer writing until training end
                self._best_state = {
                    'epoch': epoch,
                    'G_state_dict': {k: v.cpu() for k, v in self.G.state_dict().items()},
                    'D_state_dict': {k: v.cpu() for k, v in self.D.state_dict().items()},
                    'opt_G': self.opt_G.state_dict(),
                    'opt_D': self.opt_D.state_dict(),
                    'train_config': TRAIN_CONFIG,
                    'model_config': MODEL_CONFIG,
                }
                print(f"[BEST] Initial best ({self.save_criterion})={score:.6f} at epoch {epoch} (saved in-memory)")
            else:
                # save to disk immediately; this will be the first best file
                self._save_checkpoint(epoch, tag='best')
                print(f"[BEST] Initial best ({self.save_criterion})={score:.6f} at epoch {epoch}")
            return

        improved = (score < self.best_score) if self._minimize else (score > self.best_score)
        if improved:
            old = self.best_score
            prev_epoch = self.best_epoch
            self.best_score = score
            self.best_epoch = epoch

            if self.save_best_at_end:
                # update in-memory snapshot (do not write file now)
                self._best_state = {
                    'epoch': epoch,
                    'G_state_dict': {k: v.cpu() for k, v in self.G.state_dict().items()},
                    'D_state_dict': {k: v.cpu() for k, v in self.D.state_dict().items()},
                    'opt_G': self.opt_G.state_dict(),
                    'opt_D': self.opt_D.state_dict(),
                    'train_config': TRAIN_CONFIG,
                    'model_config': MODEL_CONFIG,
                }
                print(f"[BEST] Improved ({self.save_criterion}): {old:.6f} -> {score:.6f} at epoch {epoch} (in-memory)")
            else:
                # remove previous best file (if present) so only one best exists on disk
                try:
                    old_path = self._checkpoint_path(prev_epoch, 'best')
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except Exception:
                    pass

                # save new best to disk
                self._save_checkpoint(epoch, tag='best')
                print(f"[BEST] Improved ({self.save_criterion}): {old:.6f} -> {score:.6f} at epoch {epoch}")
        else:
            print(f"[BEST] No improvement ({self.save_criterion}) this epoch: {score:.6f} (best {self.best_score:.6f} @{self.best_epoch})")

    # ======================================================
    # Override by subclasses (DCGANTrainer, LSGANTrainer...)
    # ======================================================
    def train_step(self, real_batch):
        raise NotImplementedError

    # ======================================================
    # Universal training loop
    # ======================================================
    def train(self):
        epochs = TRAIN_CONFIG["epochs"]
        visualize_interval = TRAIN_CONFIG.get("visualize_interval", 5)

        for epoch in range(epochs):
            d_losses, g_losses = [], []

            for real, _ in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                d_loss, g_loss = self.train_step(real)
                d_losses.append(d_loss)
                g_losses.append(g_loss)

            d_mean = sum(d_losses) / len(d_losses) if len(d_losses) > 0 else float('nan')
            g_mean = sum(g_losses) / len(g_losses) if len(g_losses) > 0 else float('nan')

            # Save logs
            self.save_metrics([d_mean, g_mean], epoch + 1)

            # Save images
            if (epoch + 1) % visualize_interval == 0:
                self.save_samples(epoch + 1)

            # ---- Checkpoint (best/last) ----
            try:
                self._maybe_save_best_and_last(epoch + 1, d_mean, g_mean)
            except Exception as e:
                print(f"[WARN] Failed to save checkpoint at epoch {epoch+1}: {e}")

            # ---- Step LR scheduler ----
            if self.scheduler_G:
                self.scheduler_G.step()
                self.scheduler_D.step()

        # If configured to save only one best at the end, write it out now.
        if self.save_best_at_end and self._best_state is not None:
            try:
                # clean any existing best file name for consistency
                best_path = self._checkpoint_path(self.best_epoch, 'best')
                if os.path.exists(best_path):
                    os.remove(best_path)
                torch.save(self._best_state, best_path)
                print(f"[CHECKPOINT] Saved single best checkpoint at end: {best_path}")
            except Exception as e:
                print(f"[WARN] Failed to save final best checkpoint: {e}")

        self.visualize_log()

    # ======================================================
    # Folder preparation
    # ======================================================
    def _prepare_output_dir(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[INFO] Output directory reset: {self.out_dir}")

    # ======================================================
    # Sample save
    # ======================================================
    def save_samples(self, epoch, num_samples=None, nrow=None):
        num_samples = int(num_samples or TRAIN_CONFIG.get("save_num", 4))
        nrow = int(nrow or TRAIN_CONFIG.get("save_nrow", max(1, int(math.sqrt(num_samples)))))

        self.G.eval()
        with torch.no_grad():
            if self.fixed_z.size(0) >= num_samples:
                z = self.fixed_z[:num_samples]
            else:
                z = torch.randn(num_samples, self.z_dim, device=self.device)

            fake = self.G(z)
            fake = (fake + 1) / 2.0

            grid = vutils.make_grid(fake, nrow=nrow, padding=10, normalize=False)
            save_path = os.path.join(self.out_dir, f"epoch_{epoch:03d}_samples.jpg")

            plt.figure(figsize=(nrow * 2, math.ceil(num_samples / nrow) * 2))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=150)
            plt.close()

        self.G.train()
        print(f"[VISUALIZE] Saved sample to {save_path}")

    # ======================================================
    # CSV log
    # ======================================================
    def save_metrics(self, metrics_log, epoch):
        csv_path = os.path.join(self.out_dir, "training_log.csv")
        exists = os.path.exists(csv_path)

        header = ["epoch", "d_loss", "g_loss", "lr_G", "lr_D", "batch_size", "img_size", "z_dim"]

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(header)

            writer.writerow([
                epoch,
                metrics_log[0],
                metrics_log[1],
                self.opt_G.param_groups[0]["lr"],
                self.opt_D.param_groups[0]["lr"],
                TRAIN_CONFIG.get("batch_size"),
                MODEL_CONFIG.get("image_size"),
                MODEL_CONFIG.get("z_dim"),
            ])

    # ======================================================
    # Plot loss curve
    # ======================================================
    def visualize_log(self):
        import pandas as pd

        csv_path = os.path.join(self.out_dir, "training_log.csv")
        if not os.path.exists(csv_path):
            print("[WARN] No CSV log found, skip visualization.")
            return

        df = pd.read_csv(csv_path)

        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["d_loss"], label="D Loss")
        plt.plot(df["epoch"], df["g_loss"], label="G Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{MODEL_CONFIG['model_name'].upper()} Training Loss")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.out_dir, "loss_curve.jpg")
        plt.savefig(save_path)
        plt.close()

        print(f"[VISUALIZE] Saved loss curve to {save_path}")
