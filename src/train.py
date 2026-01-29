import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import json
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

def calculate_confidence_intervals(y_true, y_pred_probs, n_bootstraps=1000, alpha=0.95):
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    rng = np.random.RandomState(42)
    metrics = {'accuracy': [], 'auc': []}
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        metrics['accuracy'].append(accuracy_score(y_true[indices], y_pred[indices]))
        metrics['auc'].append(roc_auc_score(y_true[indices], y_pred_probs[indices]))
    
    results = {}
    for metric, values in metrics.items():
        lower = np.percentile(values, (1 - alpha) / 2 * 100)
        upper = np.percentile(values, (1 + alpha) / 2 * 100)
        mean = np.mean(values)
        results[metric] = f"{mean:.4f} [{lower:.4f} - {upper:.4f}]"
        
    return results

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Test Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)
            preds = (scores > 0.5).float()
            
            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.0 # Handle case with one class
        
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sensitivity = recall_score(all_labels, all_preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    cis = calculate_confidence_intervals(all_labels, all_scores)

    results = {
        "accuracy": acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confidence_intervals": cis
    }
    
    print("Test Results:", json.dumps(results, indent=2))
    
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

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
            all_scores = []

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
                    
                    scores = torch.sigmoid(outputs)
                    preds = (scores > 0.5).float()

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().detach().numpy())
                
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
            
            if phase == 'val':
                try:
                    auc = roc_auc_score(all_labels, all_scores)
                except ValueError:
                    auc = 0.0
                tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                print(f"Val AUC: {auc:.4f} Spec: {specificity:.4f}")
                print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds, labels=[0, 1])}")

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
    
    # Final Evaluation
    print("\nEvaluating on Test Set...")
    # Load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
