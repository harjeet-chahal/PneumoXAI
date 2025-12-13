import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import shap
from src.model import PneumoniaNet
from src.dataset import get_data_loaders
from src.config import BEST_MODEL_PATH
from src.explainers import get_gradcam, get_lime_explanation, get_shap_explanation, denormalize_image

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = PneumoniaNet()
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print("Loaded best model.")
    except FileNotFoundError:
        print("Best model not found, using initialized weights (random) for testing XAI flow.")
    
    model.to(device)
    model.eval()

    # Get Data
    _, _, test_loader = get_data_loaders(batch_size=8)
    inputs, labels = next(iter(test_loader))
    
    # Select a positive case (Pneumonia) preferably
    target_idx = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            target_idx = i
            break
            
    input_tensor = inputs[target_idx].to(device)
    label = labels[target_idx]
    print(f" explaining image index {target_idx}, Label: {label}")

    # 1. Grad-CAM
    print("Generating Grad-CAM...")
    gradcam_img = get_gradcam(model, input_tensor)
    cv2.imwrite('gradcam.png', cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
    
    # 2. LIME
    print("Generating LIME...")
    lime_img = get_lime_explanation(model, input_tensor)
    # LIME returns float 0-1 or uint8? implementation returns uint8
    cv2.imwrite('lime.png', cv2.cvtColor(lime_img, cv2.COLOR_RGB2BGR))
    
    # 3. SHAP
    print("Generating SHAP...")
    background = inputs[:4].to(device) # use first 4 as background
    shap_vals, shap_img = get_shap_explanation(model, input_tensor, background)
    
    # Plot using shap
    shap.image_plot(shap_vals, shap_img, show=False)
    plt.savefig('shap.png')
    plt.close()

    print("XAI Test Complete. Check gradcam.png, lime.png, shap.png")

if __name__ == '__main__':
    main()
