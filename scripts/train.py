"""
Training loop utilities.
"""
from typing import Dict, Optional
import mlflow
# from sklearn.base import r2_score
from sklearn.isotonic import spearmanr
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from scripts.plots import plot_residuals
from scripts.models import vae_nb_loss
from scripts.utils import EarlyStopping, pearson_corr, classification_error


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopper: Optional[EarlyStopping] = None,
    device: str = "cpu",
    gradient_accumulation_steps: int = 1,
    log_interval: int = 50,
    iso_to_gene_index: Optional[torch.Tensor] = None,
    use_amp: bool = True,
) -> Dict[str, list]:
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "val_corr": [], "val_class_err": [], "epoch_time": []}
    amp_enabled = use_amp and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for epoch in range(num_epochs):
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        running_loss = torch.zeros((), device=device)
        total_samples = 0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for i, (inputs, targets) in enumerate(iterator):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            batch_size = inputs.size(0)
            
            # Forward pass with optional AMP
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            
            # Normalize loss for gradient accumulation
            loss_scaled = loss / gradient_accumulation_steps
            scaler.scale(loss_scaled).backward()
            micro_step += 1
            
            # Step optimizer only every N steps
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

            # Only accumulate loss during training (metrics are expensive)
            running_loss += loss.detach() * batch_size
            total_samples += batch_size

            if (i + 1) % log_interval == 0:
                avg_loss = (running_loss / total_samples).item()
                iterator.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Flush leftover grads for last partial accumulation
        if micro_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss.item() / total_samples

        # Validation (compute all metrics here)
        model.eval()
        val_running_loss = torch.zeros((), device=device)
        val_running_corr = torch.zeros((), device=device)
        val_running_class_err = torch.zeros((), device=device)
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()
                batch_size = inputs.size(0)
                
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                
                corr = pearson_corr(outputs.flatten(), targets.flatten())
                if iso_to_gene_index is not None:
                    err = classification_error(outputs, targets, iso_to_gene_index)
                else:
                    err = float("nan")
                
                val_running_loss += loss * batch_size
                val_running_corr += corr * batch_size
                val_running_class_err += torch.tensor(err, device=device) * batch_size
                val_total += batch_size

        val_loss = val_running_loss.item() / max(1, val_total)
        val_corr = val_running_corr.item() / max(1, val_total)
        val_class_err = val_running_class_err.item() / max(1, val_total)
        
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        epoch_duration = end_time - start_time
        samples_per_sec = total_samples / epoch_duration

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_corr"].append(val_corr)
        history["val_class_err"].append(val_class_err)
        history["epoch_time"].append(epoch_duration)

        if scheduler is not None:
            if hasattr(scheduler, "step") and "metrics" in scheduler.step.__code__.co_varnames:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch+1:03d} | Time: {epoch_duration:.2f}s ({samples_per_sec:.1f} samp/s) | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_corr={val_corr:.4f} val_err={val_class_err:.4f}")

        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_corr": val_corr,
                "val_class_err": val_class_err,
                "epoch_time": epoch_duration,
                "samples_per_sec": samples_per_sec
            }, step=epoch)

        if early_stopper and early_stopper(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopper: Optional[EarlyStopping] = None,
    device: str = "cpu",
    progress: bool = True,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False,
    iso_to_gene_index: Optional[torch.Tensor] = None,
) -> Dict[str, list]:
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_corr": [], "val_corr": [], 
               "train_class_err": [], "val_class_err": [], "val_preds": [], "val_targets": []}
    amp_enabled = use_amp and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for epoch in range(num_epochs):

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_corr = 0.0
        running_class_err = 0.0
        total = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) if progress else train_loader
        for inputs, targets in iterator:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corr += pearson_corr(outputs.flatten(), targets.flatten()).item() * batch_size

            if iso_to_gene_index is not None:
                iso_to_gene_index = torch.from_numpy(iso_to_gene_index)
                running_class_err += classification_error(outputs, targets, iso_to_gene_index) * batch_size
            total += batch_size

        train_loss = running_loss / total
        train_corr = running_corr / total
        train_class_err = running_class_err / total

        # VALIDATION

        model.eval()

        val_loss = 0.0
        val_corr = 0.0
        val_class_err = 0.0
        val_total = 0
        preds_list, targets_list = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                val_corr += pearson_corr(outputs.flatten(), targets.flatten()).item() * batch_size
                if iso_to_gene_index is not None:
                    val_class_err += classification_error(outputs, targets, iso_to_gene_index) * batch_size
                val_total += batch_size
                preds_list.append(outputs.cpu())
                targets_list.append(targets.cpu())


        val_loss /= max(1, val_total)
        val_corr /= max(1, val_total)
        val_class_err /= max(1, val_total)

        preds = torch.cat(preds_list)
        targets = torch.cat(targets_list)

        history["train_loss"].append(train_loss)
        history["train_corr"].append(train_corr)
        history["train_class_err"].append(train_class_err)
        history["val_loss"].append(val_loss)
        history["val_corr"].append(val_corr)
        history["val_class_err"].append(val_class_err)
        history["val_preds"] = preds
        history["val_targets"] = targets

        # rmse = torch.sqrt(torch.mean((preds - targets) ** 2)) # RMSE is fine on logged data
        # corr_logged = pearson_corr(preds, targets) # might be problematic
        # corr_unlogged = pearson_corr(preds_unlogged, targets_unlogged) # Calculate Pearson on unlogged data
        # spearman_corr, _ = spearmanr(preds_unlogged.cpu().numpy(), targets_unlogged.cpu().numpy())
        # r2_correlation = r2_score(targets.cpu().numpy(), preds.cpu().numpy())
        
        # print(f"MLP Validation RMSE: {rmse:.4f}")
        # print(f"MLP Validation Pearson (logged): {corr_logged:.4f}")
        # print(f"MLP Validation Pearson (unlogged): {corr_unlogged:.4f}")
        # print(f"MLP Validation Spearman (unlogged): {spearman_corr:.4f}")
        # print(f"MLP Validation R2 Correlation: {r2_correlation:.4f}")

        if scheduler is not None:
            if hasattr(scheduler, "step") and "metrics" in scheduler.step.__code__.co_varnames:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_corr={train_corr:.4f} val_corr={val_corr:.4f} train_err={train_class_err:.4f} val_err={val_class_err:.4f}")
        
        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_corr": train_corr,
                "val_corr": val_corr,
                "train_class_err": train_class_err,
                "val_class_err": val_class_err,
                # "Validation_RMSE": rmse.item(), 
                # "Validation_Pearson_unlogged": corr_unlogged.item(), 
                # "Validation_Pearson_logged": corr_logged.item(),
                # "Validation_Spearman_unlogged": spearman_corr,
                # "Validation_R2_Correlation": r2_correlation
            }, step=epoch)

        if early_stopper and early_stopper(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history


def train_vae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    beta_max: float = 1.0,
    warmup_epochs: int = 3,
    early_stopper: Optional[EarlyStopping] = None,
) -> Dict[str, list]:
    """
    Train VAE with NB loss and beta-VAE style KL warm-up.
    Also evaluate on a validation set at the end of each epoch.
    """

    def _unwrap_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return batch[0]
            if len(batch) == 1:
                return batch[0]
            raise ValueError(f"Unexpected batch structure of length {len(batch)}")
        return batch

    print("Starting VAE training...")
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"beta_max: {beta_max}")
    print(f"warmup_epochs: {warmup_epochs}")

    model.to(device)
    model.train()

    # Flag to print debug shapes only once
    printed_debug_shapes = False

    # Dictionary to store training and validation metrics
    history = {
        "loss": [],              # train total loss
        "recon_loss": [],        # train reconstruction loss
        "kl_loss": [],           # train KL loss
        "val_loss": [],          # validation total loss
        "val_recon_loss": [],    # validation reconstruction loss
        "val_kl_loss": [],       # validation KL loss
        "beta": [],              # beta schedule
    }

    for epoch in range(1, num_epochs + 1):
        beta = beta_max * min(1.0, epoch / float(warmup_epochs))

        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            x_batch = _unwrap_batch(batch).to(device)
            optimizer.zero_grad()
            px_mu, px_r, mu, logvar = model(x_batch)

            if not printed_debug_shapes:
                print(f"\n[DEBUG] Shapes: x={x_batch.shape} px_mu={px_mu.shape} px_r={px_r.shape} mu={mu.shape} logvar={logvar.shape}")
                printed_debug_shapes = True

            loss, recon_loss, kl_loss = vae_nb_loss(x_batch, px_mu, px_r, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            num_train_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"[Train | Epoch {epoch} | Batch {batch_idx+1}] loss: {loss.item():.4f} | recon: {recon_loss.item():.4f} | kl: {kl_loss.item():.4f}")

        avg_loss = epoch_loss / max(1, num_train_batches)
        avg_recon = epoch_recon / max(1, num_train_batches)
        avg_kl = epoch_kl / max(1, num_train_batches)

        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x_batch = _unwrap_batch(batch).to(device)
                px_mu, px_r, mu, logvar = model(x_batch)
                val_loss, val_recon, val_kl = vae_nb_loss(x_batch, px_mu, px_r, mu, logvar, beta=beta)

                val_loss_sum += val_loss.item()
                val_recon_sum += val_recon.item()
                val_kl_sum += val_kl.item()
                num_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        avg_val_recon = val_recon_sum / max(1, num_val_batches)
        avg_val_kl = val_kl_sum / max(1, num_val_batches)

        history["loss"].append(avg_loss)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["val_loss"].append(avg_val_loss)
        history["val_recon_loss"].append(avg_val_recon)
        history["val_kl_loss"].append(avg_val_kl)
        history["beta"].append(beta)

        print(
            f"\n[Epoch {epoch}/{num_epochs}] "
            f"beta={beta:.3f} | "
            f"train_loss: {avg_loss:.4f} | "
            f"train_recon: {avg_recon:.4f} | "
            f"train_kl: {avg_kl:.4f} | "
            f"val_loss: {avg_val_loss:.4f} | "
            f"val_recon: {avg_val_recon:.4f} | "
            f"val_kl: {avg_val_kl:.4f}\n"
        )

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "train_recon": avg_recon,
                    "train_kl": avg_kl,
                    "val_loss": avg_val_loss,
                    "val_recon": avg_val_recon,
                    "val_kl": avg_val_kl,
                    "beta": beta,
                },
                step=epoch,
            )

        if early_stopper and early_stopper(avg_val_loss):
            print(
                f"[EarlyStopping] Stopped epoch {epoch}: "
                f"val_loss did not improve over last {early_stopper.patience} epochs."
            )
            break

    return history