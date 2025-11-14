import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset
from model import SuperResolutionDiffusion, SuperResDiffusionUNet, Upsampler
from losses import HybridLoss, compute_ellipticity_from_moments
import random
import time
import torch.nn.functional as F

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
        "upscale_factor": 2,
        "mse_weight": 1.0,
        "l1_weight": 0.1,
        "perceptual_weight": 0.01,
        "shape_weight": 0.1,
        "weight_decay": 0.01,
        "output_size": [66, 66]  # Make output size configurable
    }
    for key, value in default_config.items():
        if not hasattr(wandb.config, key):
            setattr(wandb.config, key, value)

def validate_model(model, val_loader, device, config):
    """Validation loop."""
    model.eval()
    val_loss = 0.0
    val_shape_errors = []
    
    with torch.no_grad():
        for lr_batch, hr_batch in val_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            t = torch.randint(0, config.timesteps, (lr_batch.shape[0],), device=device)
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                # Training mode: predict noise
                predicted_noise, noise = model(lr_batch, t, training=True)
                # Loss is between predicted and actual noise (use MSE like training)
                loss = F.mse_loss(predicted_noise, noise)
            
            val_loss += loss.item()
            
            # Compute shape errors on a sample
            if len(val_shape_errors) < 10:  # Sample a few for shape error
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    # Inference mode: get denoised output
                    t_zero = torch.zeros((lr_batch.shape[0],), dtype=torch.long, device=device)
                    sr_output = model(lr_batch, t_zero, training=False)
                    e_pred = compute_ellipticity_from_moments(sr_output)
                    e_true = compute_ellipticity_from_moments(hr_batch)
                    shape_error = torch.abs(e_pred - e_true).mean(dim=0)
                    val_shape_errors.append(shape_error)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_shape_error = torch.stack(val_shape_errors).mean(dim=0) if val_shape_errors else torch.tensor([0.0, 0.0])
    
    return avg_val_loss, avg_shape_error

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
        output_size=config.output_size if hasattr(config, 'output_size') else None
    ).to(device)

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

    scaler = torch.amp.GradScaler()
    best_loss = float("inf")

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            t = torch.randint(0, config.timesteps, (lr_batch.shape[0],), device=device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                # Training mode: predict noise
                predicted_noise, noise = model(lr_batch, t, training=True)
                # Loss is between predicted and actual noise (standard diffusion loss)
                loss = F.mse_loss(predicted_noise, noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_loss, val_shape_error = validate_model(model, val_loader, device, config)
        scheduler.step(val_loss)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_shape_error_e1": val_shape_error[0].item(),
            "val_shape_error_e2": val_shape_error[1].item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_time": time.time() - start_time
        })

        # Log images every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                random_idx = random.randint(0, len(val_loader.dataset) - 1)
                lr_img, hr_img = val_loader.dataset[random_idx]
                lr_img = lr_img.unsqueeze(0).to(device)
                t_test = torch.zeros((1,), dtype=torch.long, device=device)
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    sr_img = model(lr_img, t_test, training=False).cpu().squeeze(0)

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

        # Save best model based on validation loss
        if val_loss < best_loss - 0.001:
            best_loss = val_loss
            best_model_path = os.path.join(wandb.run.dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated (Val Loss: {best_loss:.6f})")

    final_model_path = os.path.join(wandb.run.dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Final model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()
