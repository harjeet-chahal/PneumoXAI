import torch
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PneumoniaNet

def test_model_output_shape():
    """
    Verify that the model outputs a single logit for binary classification.
    """
    model = PneumoniaNet()
    
    # Create a dummy input tensor: Batch size 1, 3 channels, 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    
    # Check shape
    # Expected: (1, 1) because we replaced fc with Linear(num_ftrs, 1)
    # Note: If we added Dropout, it's a Sequential, but output is still (N, 1)
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
    
    print("Model output shape verification passed!")
