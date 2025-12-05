"""
Dedicated LSTM training function - stable version for LSTM models.
This function will not be modified by other team members.
"""
from typing import Dict, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.utils import EarlyStopping, pearson_corr, classification_error


def train_LSTM(
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
    use_amp: bool = True,
    iso_to_gene_index: Optional[torch.Tensor] = None,
) -> Dict[str, list]:
    """
    Train an LSTM model with gradient accumulation and mixed precision.
    
    Args:
        model: LSTM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        num_epochs: Number of epochs
        scheduler: Learning rate scheduler (optional)
        early_stopper: Early stopping callback (optional)
        device: Device to train on
        progress: Show progress bar
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_amp: Use automatic mixed precision
        iso_to_gene_index: Gene index for classification error (optional)
    
    Returns:
        Dictionary with training history
    """
    model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_corr": [],
        "val_corr": [],
        "train_class_err": [],
        "val_class_err": []
    }
    
    amp_enabled = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    for epoch in range(num_epochs):
        # TRAINING
        model.train()
        optimizer.zero_grad(set_to_none=True)

        micro_step = 0
        running_loss = 0.0
        running_corr = 0.0
        running_class_err = 0.0
        total = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) if progress else train_loader
        for i, (inputs, targets) in enumerate(iterator):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            
            # Normalize loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            micro_step += 1
            
            # Step optimizer every N steps
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

            # Accumulate metrics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size * gradient_accumulation_steps
            running_corr += pearson_corr(outputs.flatten(), targets.flatten()).item() * batch_size
            if iso_to_gene_index is not None:
                running_class_err += classification_error(outputs, targets, iso_to_gene_index) * batch_size
            total += batch_size

        # Flush leftover gradients
        if micro_step > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / total
        train_corr = running_corr / total
        train_class_err = running_class_err / total

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_corr = 0.0
        val_class_err = 0.0
        val_total = 0
        
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

        val_loss /= max(1, val_total)
        val_corr /= max(1, val_total)
        val_class_err /= max(1, val_total)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_corr"].append(train_corr)
        history["train_class_err"].append(train_class_err)
        history["val_loss"].append(val_loss)
        history["val_corr"].append(val_corr)
        history["val_class_err"].append(val_class_err)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
              f"Corr: {train_corr:.4f}/{val_corr:.4f} - "
              f"ClassErr: {train_class_err:.4f}/{val_class_err:.4f}")

        # Learning rate scheduler
        if scheduler is not None:
            if hasattr(scheduler, "step") and "metrics" in scheduler.step.__code__.co_varnames:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping
        if early_stopper and early_stopper(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history
