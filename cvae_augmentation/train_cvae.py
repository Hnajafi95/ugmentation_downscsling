"""
Training script for cVAE model.

Trains a conditional variational autoencoder to generate high-resolution precipitation
conditioned on low-resolution multi-variable inputs and static maps.
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import csv
from tqdm import tqdm

from model_cvae import CVAE
from losses import CVAELoss
from losses_simplified import SimplifiedCVAELoss, MinimalCVAELoss
from utils_metrics import MetricsTracker
from data_io import get_dataloaders, load_thresholds


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model, config):
    """Create optimizer from config."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler from config."""
    sched_config = config['scheduler']
    sched_type = sched_config['type']

    if sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max'],
            eta_min=sched_config['eta_min']
        )
    elif sched_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config['factor'],
            patience=sched_config['patience']
        )
    elif sched_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    return scheduler, sched_type


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, epoch):
    """Train for one epoch."""
    model.train()
    criterion.update_beta(epoch)

    total_loss = 0.0
    total_rec = 0.0
    total_mass = 0.0
    total_kl = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        X_lr = batch['X_lr'].to(device)        # (B, C, H_lr, W_lr)
        Y_hr = batch['Y_hr'].to(device)        # (B, 1, H, W)
        S = batch['S'].to(device)              # (B, S_ch, H, W)
        mask_land = batch['mask_land'].to(device)  # (1, 1, H, W) or (B, 1, H, W)

        optimizer.zero_grad()

        # Forward pass with AMP
        if config['train']['amp']:
            with autocast():
                Y_hat, mu, logvar = model(X_lr, Y_hr, S)
                loss_dict = criterion(Y_hr, Y_hat, mu, logvar, mask_land)
                loss = loss_dict['loss']
        else:
            Y_hat, mu, logvar = model(X_lr, Y_hr, S)
            loss_dict = criterion(Y_hr, Y_hat, mu, logvar, mask_land)
            loss = loss_dict['loss']

        # Backward pass
        if config['train']['amp']:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_rec += loss_dict['L_rec'].item()
        total_mass += loss_dict['L_mass'].item()
        total_kl += loss_dict['L_kl'].item()
        n_batches += 1

        # Update progress bar
        if (batch_idx + 1) % config['validation']['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{total_loss/n_batches:.4f}",
                'rec': f"{total_rec/n_batches:.4f}",
                'kl': f"{total_kl/n_batches:.4f}"
            })

    # Average losses
    avg_loss = total_loss / n_batches
    avg_rec = total_rec / n_batches
    avg_mass = total_mass / n_batches
    avg_kl = total_kl / n_batches

    metrics = {
        'loss': avg_loss,
        'L_rec': avg_rec,
        'L_mass': avg_mass,
        'L_kl': avg_kl,
        'beta': criterion.current_beta
    }

    return metrics


def validate_epoch(model, val_loader, criterion, device, config, p95, land_mask):
    """Validate for one epoch."""
    model.eval()

    total_loss = 0.0
    total_rec = 0.0
    total_mass = 0.0
    total_kl = 0.0
    n_batches = 0

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(p95, land_mask)

    pbar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for batch in pbar:
            # Move data to device
            X_lr = batch['X_lr'].to(device)
            Y_hr = batch['Y_hr'].to(device)
            S = batch['S'].to(device)
            mask_land = batch['mask_land'].to(device)

            # Forward pass
            if config['train']['amp']:
                with autocast():
                    Y_hat, mu, logvar = model(X_lr, Y_hr, S)
                    loss_dict = criterion(Y_hr, Y_hat, mu, logvar, mask_land)
            else:
                Y_hat, mu, logvar = model(X_lr, Y_hr, S)
                loss_dict = criterion(Y_hr, Y_hat, mu, logvar, mask_land)

            # Accumulate losses
            total_loss += loss_dict['loss'].item()
            total_rec += loss_dict['L_rec'].item()
            total_mass += loss_dict['L_mass'].item()
            total_kl += loss_dict['L_kl'].item()
            n_batches += 1

            # Update metrics tracker
            metrics_tracker.update(Y_hr, Y_hat)

    # Average losses
    avg_loss = total_loss / n_batches
    avg_rec = total_rec / n_batches
    avg_mass = total_mass / n_batches
    avg_kl = total_kl / n_batches

    # Compute metrics
    metrics = metrics_tracker.compute()
    metrics.update({
        'loss': avg_loss,
        'L_rec': avg_rec,
        'L_mass': avg_mass,
        'L_kl': avg_kl,
        'beta': criterion.current_beta
    })

    # Compute combined metric if configured
    if 'combined_metric_weights' in config.get('train', {}):
        weights = config['train']['combined_metric_weights']
        combined = (weights.get('MAE_all', 0.5) * metrics['MAE_all'] +
                    weights.get('MAE_tail', 0.5) * metrics['MAE_tail'])
        metrics['combined_metric'] = combined

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics
    }
    torch.save(checkpoint, path)


def main(args):
    """Main training function."""
    print("=" * 80)
    print("cVAE Training")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")

    # Set random seed
    set_seed(config['seed'])

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    outputs_root = Path(config['outputs_root'])
    checkpoints_dir = outputs_root / "checkpoints"
    logs_dir = outputs_root / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load thresholds
    thresholds_path = Path(config['data_root']) / "data" / "metadata" / "thresholds.json"
    thresholds = load_thresholds(thresholds_path)
    p95 = thresholds['P95']
    print(f"\nLoaded thresholds: P95={p95:.4f}, P99={thresholds['P99']:.4f}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=config['data_root'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        use_land_sea=config['statics']['use_land_sea'],
        use_dist_coast=config['statics']['use_dist_coast'],
        use_elevation=config['statics']['use_elevation'],
        normalize_statics=config['statics']['normalize_statics']
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Get land mask from dataset
    sample_batch = next(iter(train_loader))
    land_mask = sample_batch['mask_land'][:1].to(device)  # (1, 1, H, W)

    # Determine number of input channels from data
    X_sample = sample_batch['X_lr']
    C = X_sample.shape[1]
    print(f"\nDetected {C} input channels in X_lr")

    # Update config with detected channels
    config['model']['in_channels_X'] = C

    # Create model
    print("\nCreating model...")
    model = CVAE(
        in_channels_X=config['model']['in_channels_X'],
        in_channels_Y=config['model']['in_channels_Y'],
        static_channels=config['model']['static_channels'],
        d_x=config['model']['d_x'],
        d_y=config['model']['d_y'],
        d_z=config['model']['d_z'],
        H=config['model']['H'],
        W=config['model']['W'],
        base_filters=config['model']['base_filters']
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Create loss criterion
    loss_type = config['loss'].get('type', 'original')

    print(f"\nCreating loss criterion: {loss_type}")

    if loss_type == 'simplified':
        # Simplified loss: ONE weighted reconstruction + mass + KL
        criterion = SimplifiedCVAELoss(
            p99=thresholds.get('P99', p95),  # Use P99 if available, else P95
            scale=config['loss'].get('scale', 36.6),
            w_min=config['loss'].get('w_min', 0.1),
            w_max=config['loss'].get('w_max', 2.0),
            tail_boost=config['loss'].get('tail_boost', 1.5),
            lambda_mass=config['loss'].get('lambda_mass', 0.01),
            beta_kl=config['loss']['beta_kl'],
            warmup_epochs=config['loss']['warmup_epochs']
        )
        print(f"  Intensity weighting: scale={config['loss'].get('scale', 36.6)}, "
              f"tail_boost={config['loss'].get('tail_boost', 1.5)}")

    elif loss_type == 'minimal':
        # Minimal loss: ONLY weighted reconstruction + KL (no mass conservation)
        criterion = MinimalCVAELoss(
            p99=thresholds.get('P99', p95),
            scale=config['loss'].get('scale', 36.6),
            tail_boost=config['loss'].get('tail_boost', 1.5),
            beta_kl=config['loss']['beta_kl'],
            warmup_epochs=config['loss']['warmup_epochs']
        )
        print(f"  Minimal loss (2 terms only): weighted_rec + KL")

    else:
        # Original multi-objective loss
        criterion = CVAELoss(
            p95=p95,
            lambda_base=config['loss']['lambda_base'],
            lambda_ext=config['loss']['lambda_ext'],
            lambda_mass=config['loss']['lambda_mass'],
            beta_kl=config['loss']['beta_kl'],
            warmup_epochs=config['loss']['warmup_epochs']
        )
        print(f"  Original loss: lambda_base={config['loss']['lambda_base']}, "
              f"lambda_ext={config['loss']['lambda_ext']}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler, sched_type = create_scheduler(optimizer, config)

    # Create AMP scaler
    scaler = GradScaler() if config['train']['amp'] else None

    # Initialize CSV log
    log_file = logs_dir / "train_log.csv"
    log_fields = ['epoch', 'train_loss', 'train_L_rec', 'train_L_mass', 'train_L_kl',
                  'val_loss', 'val_L_rec', 'val_L_mass', 'val_L_kl',
                  'val_MAE_all', 'val_MAE_tail', 'val_RMSE_all', 'val_mass_bias',
                  'beta', 'lr', 'val_combined_metric']

    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    print(f"\nLogging to: {log_file}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    # Get early stopping metric from config (default to MAE_all for stability)
    early_stopping_metric = config['train'].get('early_stopping_metric', 'MAE_all')
    print(f"Early stopping metric: {early_stopping_metric}")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config['train']['epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, epoch
        )

        # Validate
        if (epoch + 1) % config['validation']['val_every'] == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, config, p95, land_mask
            )

            # Print metrics
            print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Rec: {train_metrics['L_rec']:.4f}, "
                  f"KL: {train_metrics['L_kl']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Rec: {val_metrics['L_rec']:.4f}, "
                  f"MAE: {val_metrics['MAE_all']:.4f}, "
                  f"Tail: {val_metrics['MAE_tail']:.4f}")

            # Log to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_L_rec': train_metrics['L_rec'],
                    'train_L_mass': train_metrics['L_mass'],
                    'train_L_kl': train_metrics['L_kl'],
                    'val_loss': val_metrics['loss'],
                    'val_L_rec': val_metrics['L_rec'],
                    'val_L_mass': val_metrics['L_mass'],
                    'val_L_kl': val_metrics['L_kl'],
                    'val_MAE_all': val_metrics['MAE_all'],
                    'val_MAE_tail': val_metrics['MAE_tail'],
                    'val_RMSE_all': val_metrics['RMSE_all'],
                    'val_mass_bias': val_metrics['mass_bias'],
                    'beta': val_metrics['beta'],
                    'lr': optimizer.param_groups[0]['lr'],
                    'val_combined_metric': val_metrics.get('combined_metric', '')
                })

            # Check for improvement using configured metric
            current_metric_value = val_metrics[early_stopping_metric]
            if current_metric_value < best_val_loss:
                best_val_loss = current_metric_value
                epochs_no_improve = 0

                # Save best checkpoint
                best_path = checkpoints_dir / "cvae_best.pt"
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_path)
                print(f"  Saved best model ({early_stopping_metric}={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= config['train']['early_stopping_patience']:
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {epochs_no_improve} epochs)")
                break

        # Save periodic checkpoint
        if (epoch + 1) % config['train']['save_every'] == 0:
            ckpt_path = checkpoints_dir / f"cvae_epoch_{epoch+1:03d}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")

        # Update scheduler
        if sched_type == 'plateau':
            scheduler.step(val_metrics['loss'] if (epoch + 1) % config['validation']['val_every'] == 0 else train_metrics['loss'])
        else:
            scheduler.step()

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation {early_stopping_metric}: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoints_dir}")
    print(f"Training log saved to: {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cVAE for precipitation generation")
    parser.add_argument("--config", type=str,
                        default="/home/user/ugmentation_downscsling/cvae_augmentation/config.yaml",
                        help="Path to config file")

    args = parser.parse_args()
    main(args)
