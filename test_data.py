import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import get_data_loaders
from src.config import BATCH_SIZE

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

def main():
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=8)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get a batch of training data
    inputs, classes = next(iter(train_loader))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    classes_list = ['NORMAL' if x == 0 else 'PNEUMONIA' for x in classes]
    print(f"Batch classes: {classes_list}")

    plt.figure(figsize=(12, 4))
    imshow(out, title=str(classes_list))
    output_path = 'data_loader_test_grid.png'
    plt.savefig(output_path)
    print(f"Saved visualization grid to {output_path}")

if __name__ == '__main__':
    main()
