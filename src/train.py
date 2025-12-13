import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score
import copy
import argparse
import time
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH, TRAIN_DIR, VAL_DIR, BEST_MODEL_PATH
from src.dataset import get_data_loaders
from src.model import PneumoniaNet

# Ensure BEST_MODEL_PATH is defined if not present in config
if not 'BEST_MODEL_PATH' in locals():
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')

def train_model(model, dataloaders, dataset_sizes, device, num_epochs=25, dry_run=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.model.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            all_preds = []
            all_labels = []

            # Iterate over data.
            # Use limited batches for dry run
            batch_limit = 2 if dry_run else float('inf')
            
            pbar = tqdm(dataloaders[phase], desc=f"{phase} Phase")
            for i, (inputs, labels) in enumerate(pbar):
                if i >= batch_limit:
                    break
                    
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update pbar description
                pbar.set_postfix({'loss': loss.item()})

            if dry_run:
                epoch_loss = running_loss / (batch_limit * 8) # approx
            else:
                epoch_loss = running_loss / dataset_sizes[phase]
            
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f}')

            # Deep copy the model and Early Stopping check
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"Validation loss improved. Saved model to {BEST_MODEL_PATH}")
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

        print()

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run for limited batches for debugging')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    train_loader, val_loader, test_loader = get_data_loaders()
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Calculate dataset sizes (approximated for weighted sampler just by using len(loader.dataset))
    # Note: WeightedRandomSampler means epoch length is len(dataset)
    dataset_sizes = {
        'train': len(train_loader.dataset),
        'val': len(val_loader.dataset)
    }

    model = PneumoniaNet()
    model = model.to(device)

    train_model(model, dataloaders, dataset_sizes, device, num_epochs=EPOCHS, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
