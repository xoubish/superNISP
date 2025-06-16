import torch
from claude_model_NIR import EuclidToJWSTSuperResolution, EuclidToJWSTDataset, train_two_stage

# Paths to your numpy files
nir_train_path = '../data/euclid_NIR_cosmos_41px_Y.npy'
jwst_train_path = '../data/jwst_cosmos_205px_F115W.npy'

# Train the model
# Use the optimized version
trained_model = train_two_stage(
    nir_train_path, 
    jwst_train_path, 
    val_split=0.2,
    batch_size=16,  # Increased from 8
    num_epochs_stage1=30,  # Reduced from 100
    num_epochs_stage2=20   # Reduced from 100
)

# Save the final model
torch.save(trained_model.state_dict(), 'claude_model_NIR.pth')
print("Training completed and model saved!")