import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling.
    Model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def calibrate(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        Fails if the number of classes is not 1 (binary BCE check needed).
        """
        self.to(next(self.model.parameters()).device)
        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        # Collect all logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(next(self.model.parameters()).device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
        
        logits = torch.cat(logits_list).to(next(self.model.parameters()).device)
        labels = torch.cat(labels_list).to(next(self.model.parameters()).device).float().unsqueeze(1)

        # Calculate NLL before scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        print(f'Before temperature - NLL: {before_temperature_nll:.3f}')

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)

        # Calculate NLL after scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        print(f'Optimal temperature: {self.temperature.item():.3f}')
        print(f'After temperature - NLL: {after_temperature_nll:.3f}')

        return self


def predict_with_uncertainty(model, image_tensor, n_passes=20):
    """
    Perform Monte Carlo Dropout to estimate uncertainty.
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (1, C, H, W)
        n_passes: Number of forward passes to run
        
    Returns:
        mean_pred: Mean probability (0-1)
        uncertainty: Standard deviation of probabilities
    """
    # Enable dropout
    model.train()
    
    preds = []
    
    # We want to enable dropout but keep Batch Norm in eval mode likely?
    # Actually model.train() enables both. 
    # For fine-tuned ResNet, BN stats should probably be frozen or kept as is.
    # If the model was trained with frozen BN (which it was in model.py), model.train() might unfreeze it if not carefully handled?
    # src/model.py only freezes parameters, doesn't set mode.
    # But usually evaluating with BN in train mode on single image is bad (batch size 1).
    # Correct approach for MC Dropout: Set model to eval, then manually enable dropout layers.
    
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(image_tensor.unsqueeze(0))
            prob = torch.sigmoid(output).item()
            preds.append(prob)
            
    preds = np.array(preds)
    mean_pred = np.mean(preds)
    uncertainty = np.std(preds)
    
    return mean_pred, uncertainty
