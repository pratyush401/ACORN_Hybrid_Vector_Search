import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

torch.manual_seed(42) # Set deterministic behavior for PyTorch

# --------------------------------------------------------------
# Select compute device:
# - Prefer Apple MPS (Metal Performance Shaders) when available
# - Fall back to CPU otherwise
# --------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------
# Standard ImageNet preprocessing applied to each input image:
# - Resize to (224,224)
# - Convert to tensor
# - Normalize to match pretrained ResNet expectations
# --------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------------------
# Custom dataset to read images from a directory structure.
# Expects image files (*.jpg) inside nested folders.
# --------------------------------------------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        # Recursively gather all .jpg images
        self.image_paths = list(self.image_dir.rglob("*.jpg"))
        self.transform = transform

    def __len__(self):
        # Number of discovered images
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and apply transformations
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Return processed image + filename string
        return image, str(path.name)

# --------------------------------------------------------------
# Load pretrained ResNet50 and convert it to a feature extractor.
# Removing the classification head outputs a 2048-dim embedding.
# --------------------------------------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the final fully-connected layer to use the model as an encoder
model.eval().to(device)

# Generate embeddings
embeddings = []
filenames = []

# --------------------------------------------------------------
# Case 1: If a directory is passed as a command-line argument,
#         only process images inside that directory.
#
# Case 2: Otherwise, process ABO-style folders named "0a", "3f"
#         across multiple directory patterns.
# --------------------------------------------------------------
if len(sys.argv) > 1:
    # -------------------------
    # USER-SUPPLIED IMAGE FOLDER
    # -------------------------
    dataset = ImageFolderDataset(sys.argv[1], transform)
    print(len(dataset), "images found.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch, names in tqdm(dataloader, desc="Embedding images"):
            batch = batch.to(device)
            # Forward pass → extract embedding
            feats = model(batch).squeeze()  # shape [batch, 2048]
            feats = feats.cpu().numpy()
            embeddings.append(feats)
            filenames.extend(names)
else:
    # -----------------------------------
    # DEFAULT MODE:
    # Loop through directories 0-4 and 0-9,a-f within each
    # Example folders: "./00", "./01", "./0a", "./1c", etc.
    # -----------------------------------
    for k in [0, 1, 2, 3, 4]:
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']:
            # Skip known invalid combination in ABO structure
            if k == 0 and i == 'f':
                continue
            dataset = ImageFolderDataset(f"./{k}{i}", transform)
            print(len(dataset), "images found.")
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
                for batch, names in tqdm(dataloader, desc="Embedding images"):
                    batch = batch.to(device)
                    # Extract visual feature embedding
                    feats = model(batch).squeeze()  # shape [batch, 2048]
                    feats = feats.cpu().numpy()
                    embeddings.append(feats)
                    filenames.extend(names)
# # Stack all embeddings into one big array
# embeddings = np.vstack(embeddings)
# print(f"Embeddings shape: {embeddings.shape}")  # (N, 2048)

# --------------------------------------------------------------
# Summary stats
# --------------------------------------------------------------
print(f"Total embeddings: {len(embeddings)}")
print(f"Total filenames: {len(filenames)}")

# --------------------------------------------------------------
# Save embeddings + filenames as .npy files
#
# Note:
#   embeddings is a Python list of 2048-d vectors → converted to ndarray.
# --------------------------------------------------------------
np.save("embeddings_query_full_query.npy", np.array(embeddings))
np.save("filenames_query_full_query.npy", np.array(filenames))
print(f"Saved {len(filenames)} embeddings")
