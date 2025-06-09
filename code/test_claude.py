import torch
from claude_model_NIR import EuclidToJWSTSuperResolution, EuclidToJWSTDataset, train_two_stage

# Paths to your numpy files
nir_train_path = '../data/euclid_NIR_cosmos_41px_Y.npy'
jwst_train_path = '../data/jwst_cosmos_205px_F115W.npy'

# Train the model
trained_model = train_two_stage(
    nir_train_path, 
    jwst_train_path, 
    val_split=0.2,      # 20% validation split
    batch_size=8,       # Adjust based on GPU memory
    num_epochs_stage1=100,  # Epochs in first training stage
    num_epochs_stage2=100   # Epochs in fine-tuning stage
)

# Save the final model
torch.save(trained_model.state_dict(), 'claude_model_NIR.pth')
print("Training completed and model saved!")