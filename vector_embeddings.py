import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

torch.manual_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# # Custom dataset to load images
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.rglob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(path.name)

# # Load pretrained ResNet50 and remove classification head
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval().to(device)

# Generate embeddings
embeddings = []
filenames = []

if len(sys.argv) > 1:
    dataset = ImageFolderDataset(sys.argv[1], transform)
    print(len(dataset), "images found.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch, names in tqdm(dataloader, desc="Embedding images"):
            batch = batch.to(device)
            feats = model(batch).squeeze()  # shape [batch, 2048]
            feats = feats.cpu().numpy()
            embeddings.append(feats)
            filenames.extend(names)
else:
    for k in [0, 1, 2, 3, 4]:
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']:
            if k == 0 and i == 'f':
                continue
            dataset = ImageFolderDataset(f"./{k}{i}", transform)
            print(len(dataset), "images found.")
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
                for batch, names in tqdm(dataloader, desc="Embedding images"):
                    batch = batch.to(device)
                    feats = model(batch).squeeze()  # shape [batch, 2048]
                    feats = feats.cpu().numpy()
                    embeddings.append(feats)
                    filenames.extend(names)
# # Stack all embeddings into one big array
# embeddings = np.vstack(embeddings)
# print(f"Embeddings shape: {embeddings.shape}")  # (N, 2048)

print(f"Total embeddings: {len(embeddings)}")
print(f"Total filenames: {len(filenames)}")

# # Save results
np.save("embeddings_query_full_query.npy", np.array(embeddings))
np.save("filenames_query_full_query.npy", np.array(filenames))
print(f"✅ Saved {len(filenames)} embeddings")
