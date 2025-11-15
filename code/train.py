import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset
from model import SuperResolutionDiffusion, SuperResDiffusionUNet, Upsampler, cosine_schedule
from losses import HybridLoss, compute_ellipticity_from_moments, gaussian_weight_map
import random
import time
import torch.nn.functional as F
import copy

def setup_config_defaults():
    default_config = {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "activation_function": "ReLU",
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 64,
        "timesteps": 500,
        "inference_timesteps": 50,
        "upscale_factor": 3,
        "mse_weight": 1.0,
        "l1_weight": 0.1,
        "perceptual_weight": 0.01,
        "shape_weight": 0.15,  # Always use shape loss to preserve galaxy structure
        "hybrid_loss_weight": 0.5,  # Increased weight for hybrid loss component
        "direct_recon_weight": 0.2,  # Direct reconstruction loss weight (LR→HR supervision)
        "weight_sigma": 0.3,  # Gaussian weight map sigma (adaptive to galaxy size)
        "weight_decay": 0.01,
        "output_size": [66, 66],
        "grad_clip": 1.0,
        "warmup_epochs": 5,
        "checkpoint_interval": 20,
        "ema_decay": 0.9999,  # EMA decay rate
        "early_stop_patience": 15,  # Early stopping patience
        "early_stop_min_delta": 0.0001  # Minimum change to qualify as improvement
    }
    for key, value in default_config.items():
        if not hasattr(wandb.config, key):
            setattr(wandb.config, key, value)

def validate_model(model, val_loader, device, config, criterion):
    """Validation loop with hybrid loss."""
    model.eval()
    val_noise_loss = 0.0
    val_hybrid_loss = 0.0
    val_total_loss = 0.0
    val_shape_errors_e1 = []
    val_shape_errors_e2 = []
    
    with torch.no_grad():
        for lr_batch, hr_batch in val_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            t = torch.randint(0, config.timesteps, (lr_batch.shape[0],), device=device)
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                # Training mode: predict noise
                predicted_noise, noise = model(lr_batch, t, training=True, hr_target=hr_batch)
                
                # Hybrid loss (same as training)
                alpha_t = cosine_schedule(t, config.timesteps)
                if alpha_t.dim() == 1:
                    alpha_t = alpha_t.view(-1, 1, 1, 1)
                
                upscaled = model.upsampler(lr_batch)
                if hr_batch.shape[2:] != upscaled.shape[2:]:
                    hr_batch_resized = F.interpolate(hr_batch, size=upscaled.shape[2:], mode="bilinear", align_corners=True)
                else:
                    hr_batch_resized = hr_batch
                
                noisy_hr = alpha_t * hr_batch_resized + (1 - alpha_t) * noise
                pred_clean = (noisy_hr - (1 - alpha_t) * predicted_noise) / (alpha_t + 1e-8)
                
                # Apply adaptive Gaussian weight map centered on galaxy
                weight_sigma = getattr(config, 'weight_sigma', 0.3)
                weights = gaussian_weight_map(
                    pred_clean.shape, 
                    sigma=weight_sigma, 
                    device=pred_clean.device,
                    center_img=hr_batch_resized
                )
                
                # Weight the noise loss
                noise_weights = weights.squeeze(1) if weights.shape[1] == 1 else weights.mean(dim=1, keepdim=True).squeeze(1)
                weighted_noise_diff = (predicted_noise - noise) ** 2
                noise_loss = (weighted_noise_diff * noise_weights).mean()
                
                # Weight the hybrid loss
                pred_weighted = pred_clean * weights
                target_weighted = hr_batch_resized * weights
                
                hybrid_loss = criterion(pred_weighted, target_weighted)
                
                total_loss = noise_loss + config.hybrid_loss_weight * hybrid_loss
            
            val_noise_loss += noise_loss.item()
            val_hybrid_loss += hybrid_loss.item()
            val_total_loss += total_loss.item()
            
            # Compute shape errors on a sample
            if len(val_shape_errors_e1) < 10:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    t_zero = torch.zeros((lr_batch.shape[0],), dtype=torch.long, device=device)
                    sr_output = model(lr_batch, t_zero, training=False)
                    e_pred = compute_ellipticity_from_moments(sr_output)
                    e_true = compute_ellipticity_from_moments(hr_batch)
                    shape_error = torch.abs(e_pred - e_true).mean(dim=0)
                    val_shape_errors_e1.append(shape_error[0].item())
                    val_shape_errors_e2.append(shape_error[1].item())
    
    avg_noise_loss = val_noise_loss / len(val_loader)
    avg_hybrid_loss = val_hybrid_loss / len(val_loader)
    avg_total_loss = val_total_loss / len(val_loader)
    avg_shape_error_e1 = sum(val_shape_errors_e1) / len(val_shape_errors_e1) if val_shape_errors_e1 else 0.0
    avg_shape_error_e2 = sum(val_shape_errors_e2) / len(val_shape_errors_e2) if val_shape_errors_e2 else 0.0
    
    return avg_total_loss, avg_noise_loss, avg_hybrid_loss, (avg_shape_error_e1, avg_shape_error_e2)

def get_lr_with_warmup(optimizer, epoch, warmup_epochs, base_lr):
    """Learning rate with warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer.param_groups[0]['lr']

def update_ema(ema_model, model, decay):
    """Update EMA model weights."""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def main():
    # Initialize Weights & Biases
    wandb.init(project="super-resolution-diffusion", config={"entity": "your_wandb_entity_name"})
    setup_config_defaults()
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    activation_fn = getattr(nn, config.activation_function)

    unet = SuperResDiffusionUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_dim=config.hidden_dim,
        activation_fn=activation_fn
    ).to(device)

    upsampler = Upsampler(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        upscale_factor=config.upscale_factor
    ).to(device)

    model = SuperResolutionDiffusion(
        unet, 
        upsampler, 
        timesteps=config.timesteps,
        output_size=config.output_size if hasattr(config, 'output_size') else None,
        inference_timesteps=config.inference_timesteps if hasattr(config, 'inference_timesteps') else None
    ).to(device)

    # Create EMA model
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_loader = DataLoader(
        SuperResolutionDataset("../data/Nisp_train_cosmos.hdf5", "../data/Nircam_train_cosmos.hdf5", split="train", sample_fraction=0.2),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        SuperResolutionDataset("../data/Nisp_train_cosmos.hdf5", "../data/Nircam_train_cosmos.hdf5", split="test", sample_fraction=0.1),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    criterion = HybridLoss(
        mse_weight=config.mse_weight,
        l1_weight=config.l1_weight,
        perceptual_weight=config.perceptual_weight,
        shape_weight=config.shape_weight
    ).to(device)

    scaler = torch.amp.GradScaler()
    best_loss = float("inf")
    patience_counter = 0
    epoch_noise_loss = 0.0
    epoch_hybrid_loss = 0.0
    epoch_direct_recon_loss = 0.0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_noise_loss = 0.0
        epoch_hybrid_loss = 0.0
        start_time = time.time()

        # Learning rate warmup
        if epoch < config.warmup_epochs:
            get_lr_with_warmup(optimizer, epoch, config.warmup_epochs, config.learning_rate)

        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            t = torch.randint(0, config.timesteps, (lr_batch.shape[0],), device=device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                # Training mode: predict noise
                predicted_noise, noise = model(lr_batch, t, training=True, hr_target=hr_batch)
                
                # Additional loss: Predict clean image and use HybridLoss
                alpha_t = cosine_schedule(t, config.timesteps)
                if alpha_t.dim() == 1:
                    alpha_t = alpha_t.view(-1, 1, 1, 1)
                
                upscaled = model.upsampler(lr_batch)
                if hr_batch.shape[2:] != upscaled.shape[2:]:
                    hr_batch_resized = F.interpolate(hr_batch, size=upscaled.shape[2:], mode="bilinear", align_corners=True)
                else:
                    hr_batch_resized = hr_batch
                
                noisy_hr = alpha_t * hr_batch_resized + (1 - alpha_t) * noise
                pred_clean = (noisy_hr - (1 - alpha_t) * predicted_noise) / (alpha_t + 1e-8)
                
                # Apply adaptive Gaussian weight map centered on galaxy (use HR target to find centroid)
                weight_sigma = getattr(config, 'weight_sigma', 0.3)
                weights = gaussian_weight_map(
                    pred_clean.shape, 
                    sigma=weight_sigma, 
                    device=pred_clean.device,
                    center_img=hr_batch_resized  # Center weight on actual galaxy location
                )
                
                # Weight the noise loss to focus on galaxy region
                noise_weights = weights.squeeze(1) if weights.shape[1] == 1 else weights.mean(dim=1, keepdim=True).squeeze(1)
                weighted_noise_diff = (predicted_noise - noise) ** 2
                noise_loss = (weighted_noise_diff * noise_weights).mean()
                
                # Weight the hybrid loss
                pred_weighted = pred_clean * weights
                target_weighted = hr_batch_resized * weights
                
                hybrid_loss = criterion(pred_weighted, target_weighted)
                
                # Direct reconstruction loss: predict clean image at t=0 for stronger supervision
                # This gives the model a direct signal about the final output
                # At t=0, add tiny noise and predict it, then reconstruct clean image
                t_zero = torch.zeros_like(t)
                tiny_noise = torch.randn_like(upscaled, device=upscaled.device) * 0.01  # Very small noise
                noisy_upscaled = upscaled + tiny_noise
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    # Predict the tiny noise
                    predicted_tiny_noise = model.diffusion(noisy_upscaled, t_zero, upscaled)
                    if predicted_tiny_noise.shape != noisy_upscaled.shape:
                        predicted_tiny_noise = F.interpolate(predicted_tiny_noise, size=noisy_upscaled.shape[2:], mode="bilinear", align_corners=True)
                    
                    # Reconstruct clean image: remove predicted noise
                    direct_pred = noisy_upscaled - predicted_tiny_noise
                    if direct_pred.shape != hr_batch_resized.shape:
                        direct_pred = F.interpolate(direct_pred, size=hr_batch_resized.shape[2:], mode="bilinear", align_corners=True)
                    
                    # Weight the direct reconstruction loss
                    direct_pred_weighted = direct_pred * weights
                    direct_recon_loss = criterion(direct_pred_weighted, target_weighted)
                
                # Combined loss: weighted noise prediction + weighted image loss + direct reconstruction
                direct_recon_weight = getattr(config, 'direct_recon_weight', 0.1)
                loss = noise_loss + config.hybrid_loss_weight * hybrid_loss + direct_recon_weight * direct_recon_loss
            
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip if hasattr(config, 'grad_clip') else 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA model
            update_ema(ema_model, model, config.ema_decay)
            
            epoch_loss += loss.item()
            epoch_noise_loss += noise_loss.item()
            epoch_hybrid_loss += hybrid_loss.item()
            epoch_direct_recon_loss += direct_recon_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_noise_loss = epoch_noise_loss / len(train_loader)
        avg_hybrid_loss = epoch_hybrid_loss / len(train_loader)
        avg_direct_recon_loss = epoch_direct_recon_loss / len(train_loader)
        
        # Validation
        val_loss, val_noise_loss, val_hybrid_loss, (avg_shape_error_e1, avg_shape_error_e2) = validate_model(
            model, val_loader, device, config, criterion
        )
        scheduler.step(val_loss)

        # Log all loss components
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_noise_loss": avg_noise_loss,
            "train_hybrid_loss": avg_hybrid_loss,
            "train_direct_recon_loss": avg_direct_recon_loss,
            "val_loss": val_loss,
            "val_noise_loss": val_noise_loss,
            "val_hybrid_loss": val_hybrid_loss,
            "val_shape_error_e1": avg_shape_error_e1,
            "val_shape_error_e2": avg_shape_error_e2,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_time": time.time() - start_time
        })

        # Log images every 10 epochs (using EMA model for better quality)
        if (epoch + 1) % 10 == 0:
            ema_model.eval()
            with torch.no_grad():
                random_idx = random.randint(0, len(val_loader.dataset) - 1)
                lr_img, hr_img = val_loader.dataset[random_idx]
                lr_img = lr_img.unsqueeze(0).to(device)
                t_test = torch.zeros((1,), dtype=torch.long, device=device)
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    sr_img = ema_model(lr_img, t_test, training=False).cpu().squeeze(0)

                sr_img_tensor = sr_img.unsqueeze(0)
                hr_img_tensor = hr_img.unsqueeze(0)
                e_pred = compute_ellipticity_from_moments(sr_img_tensor)
                e_true = compute_ellipticity_from_moments(hr_img_tensor)
                shape_error = torch.abs(e_pred - e_true).squeeze()

                wandb.log({
                    "low_res": wandb.Image(lr_img.cpu().squeeze(0), caption=f"Low-Res {random_idx}"),
                    "super_res": wandb.Image(sr_img, caption=f"Super-Res {random_idx}"),
                    "high_res": wandb.Image(hr_img, caption=f"High-Res {random_idx}"),
                    "shape_error_e1": shape_error[0].item(),
                    "shape_error_e2": shape_error[1].item(),
                })

            model.train()

        # Early stopping
        if val_loss < best_loss - config.early_stop_min_delta:
            best_loss = val_loss
            patience_counter = 0
            # Save best model (using EMA model)
            best_model_path = os.path.join(wandb.run.dir, "best_model.pth")
            torch.save(ema_model.state_dict(), best_model_path)
            print(f"Best model updated (Val Loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Periodic checkpoint saving
        if hasattr(config, 'checkpoint_interval') and (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(wandb.run.dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    # Save final model (EMA)
    final_model_path = os.path.join(wandb.run.dir, "final_model.pth")
    torch.save(ema_model.state_dict(), final_model_path)
    print("Final model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()
