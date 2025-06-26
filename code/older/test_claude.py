import torch
from claude_model_NIR import StableEuclidToJWSTSuperResolution, StableFastEuclidToJWSTDataset, train_two_stage_stable

# Paths to your numpy files
nir_train_path = '../data/euclid_NIR_cosmos_41px_Y.npy'
jwst_train_path = '../data/jwst_cosmos_205px_F115W.npy'

# Stable training
trained_model = train_two_stage_stable(
    nir_train_path, 
    jwst_train_path, 
    val_split=0.2,
    batch_size=12,         # Reduced batch size
    num_epochs_stage1=30,
    num_epochs_stage2=20,
    use_amp=True
)

# Save the final model
torch.save(trained_model.state_dict(), 'claude_model_NIR_stable.pth')
print("Stable training completed and model saved!")