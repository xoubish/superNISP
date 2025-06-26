import wandb
import torch
from claudemodel import train_two_stage
from datetime import datetime

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(
        project="RRDB-twostage",
        name=f"RRDB_Euclid2JWST_5x_{timestamp}",
        config={
            "batch_size": 64,
            "num_epochs_stage1": 50,
            "num_epochs_stage2": 50,
            "sample_fraction": 1,
            "val_split": 0.2,
            "lr_data_path": "../data/Nisp_train_cosmos.hdf5",
            "hr_data_path": "../data/Nircam_train_cosmos.hdf5",
            "use_amp": False
        }
    )

    config = wandb.config

    model = train_two_stage(
        lr_hdf5_path=config.lr_data_path,
        hr_hdf5_path=config.hr_data_path,
        val_split=config.val_split,
        batch_size=config.batch_size,
        num_epochs_stage1=config.num_epochs_stage1,
        num_epochs_stage2=config.num_epochs_stage2,
        use_amp=config.use_amp,
        sample_fraction=config.sample_fraction
    )

    model_path = f"{wandb.run.dir}/final_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    wandb.finish()
