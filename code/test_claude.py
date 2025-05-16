import numpy as np
import pandas as pd
from claude_loader import create_data_loaders
from claude_model import MER2JWSTSuperResolution, train_model

# Create data loaders with flux-preserving normalization
train_loader, val_loader = create_data_loaders(
    '../data/euclid_MER_cosmos_41px_Y.npy',
    '../data/jwst_cosmos_69px_F115W.npy',
    val_split=0.2,
    batch_size=8,
    normalize_method='flux_preserving',
    seed=42
)

# Train the model
model = MER2JWSTSuperResolution()
trained_model = train_model(model, train_loader, val_loader, num_epochs=100)

torch.save(trained_model.state_dict(), "claude_model.pth")