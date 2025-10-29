import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

def main():
    print(torch.__version__)
    print(torch.backends.mps.is_available())
    

if __name__ == "__main__":
    main()