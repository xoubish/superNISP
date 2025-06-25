import wandb
from claudemodel import train_two_stage  # replace with actual filename (without .py)
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize wandb
wandb.init(project="RRDB-twostage", 
           name="RRDB_Euclid2JWST_5x",
           config={"batch_size": 2,
                   "num_epochs_stage1": 50,
                   "num_epochs_stage2": 50,
                   "sample_fraction": 1,
                   "val_split": 0.2,
                   "lr_data_path": "../data/Nisp_train_cosmos.hdf5",
                   "hr_data_path": "../data/Nircam_train_cosmos.hdf5",
                   "use_amp": False
                  })

config = wandb.config

# Call training function
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

# Save final model
model_path = f"{wandb.run.dir}/final_model_{timestamp}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

wandb.finish()
