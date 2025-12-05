"""
Plotting helpers.
"""
import matplotlib.pyplot as plt
import torch
import os


def plot_history(history: dict, out_path) -> None:
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get("train_loss", []), label="train")
    plt.plot(epochs, history.get("val_loss", []), label="val")
    plt.ylabel("MSE")
    plt.legend()
    plt.subplot(1, 2, 2)
    if "train_corr" in history:
        plt.plot(epochs, history["train_corr"], label="train")
    plt.plot(epochs, history.get("val_corr", []), label="val")
    plt.ylabel("Pearson")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_residuals(preds: torch.Tensor, targets: torch.Tensor, out_path) -> None:
    preds = preds.flatten()
    targets = targets.flatten()
    residuals = targets - preds
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(residuals.cpu().numpy(), bins=80, alpha=0.7)
    plt.title("Residuals")
    plt.subplot(1, 2, 2)
    plt.scatter(preds.cpu().numpy(), targets.cpu().numpy(), s=5, alpha=0.2)
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.title("Pred vs Target")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_vae_training(history, out_dir=None):
    """
    Plot VAE training curves:
    - train vs validation total loss
    - train vs validation reconstruction loss
    - train vs validation KL loss
    - beta schedule
    """
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["loss"], label="train total loss")
    if "val_loss" in history and len(history["val_loss"]) > 0:
        plt.plot(epochs, history["val_loss"], label="val total loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        loss_path = os.path.join(out_dir, "vae_losses.png")
        plt.savefig(loss_path)
        print(f"[plot] Saved loss curves to: {loss_path}")

    plt.figure()
    plt.plot(epochs, history["recon_loss"], label="train recon loss")
    if "val_recon_loss" in history and len(history["val_recon_loss"]) > 0:
        plt.plot(epochs, history["val_recon_loss"], label="val recon loss")
    plt.plot(epochs, history["kl_loss"], label="train KL loss")
    if "val_kl_loss" in history and len(history["val_kl_loss"]) > 0:
        plt.plot(epochs, history["val_kl_loss"], label="val KL loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()

    if out_dir is not None:
        kl_path = os.path.join(out_dir, "vae_recon_kl.png")
        plt.savefig(kl_path)
        print(f"[plot] Saved recon/KL curves to: {kl_path}")

    plt.figure()
    plt.plot(epochs, history["beta"], label="beta")
    plt.xlabel("epoch")
    plt.ylabel("beta")
    plt.legend()
    plt.tight_layout()

    if out_dir is not None:
        beta_path = os.path.join(out_dir, "vae_beta.png")
        plt.savefig(beta_path)
        print(f"[plot] Saved beta schedule to: {beta_path}")