import torch
import torch.nn.functional as F
import numpy as np
import cv2
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import copy

# Normalization constants
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denormalize_image(image_tensor):
    """
    Convert a normalized tensor to a standard numpy image (0-255).
    Input: Tensor (3, H, W)
    Output: Numpy Array (H, W, 3) in uint8
    """
    img = image_tensor.cpu().numpy().transpose((1, 2, 0))
    img = STD * img + MEAN
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

# --- Grad-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output
        output.requires_grad_(True)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.eval()
        output = self.model(x)
        
        # Binary classification: output is 1 neuron (logit)
        # We want to explain the "PNEUMONIA" class (which is likely class 1 if 0=Normal)
        # Assuming model outputs logit for class 1.
        score = output[:, 0]
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.data

def get_gradcam(model, input_tensor, target_layer=None):
    """
    Generates Grad-CAM heatmap overlay.
    """
    if target_layer is None:
        # Default to last layer of ResNet50
        target_layer = model.model.layer4[-1]

    grad_cam = GradCAM(model, target_layer)
    mask = grad_cam(input_tensor.unsqueeze(0)) # Add batch dim
    
    # Resize mask to image size
    mask = F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)
    mask = mask.squeeze().cpu().numpy()
    
    # Process original image
    orig_img = denormalize_image(input_tensor)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    superimposed_img = heatmap * 0.4 + orig_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img, mask

def calculate_iou_attention(gradcam_map, lime_mask, threshold=0.5):
    """
    Computes Intersection over Union (IoU) between Grad-CAM and LIME.
    gradcam_map: (H, W) float [0-1]
    lime_mask: (H, W) binary [0, 1]
    """
    # Binarize Grad-CAM
    gradcam_binary = (gradcam_map > threshold).astype(int)
    lime_binary = lime_mask.astype(int)
    
    intersection = np.logical_and(gradcam_binary, lime_binary)
    union = np.logical_or(gradcam_binary, lime_binary)
    
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-7)
    return iou_score

def sanity_check_gradcam(original_model, input_tensor, target_layer=None):
    """
    Performs the Randomization Test (Sanity Check) for Grad-CAM.
    Compares Grad-CAM from the trained model vs. a model with randomized weights.
    High SSIM -> Check Failed (XAI is independent of weights).
    Low SSIM -> Check Passed (XAI relies on learned weights).
    
    Returns: ssim_score, perturbed_heatmap_img
    """
    # 1. Get Original Grad-CAM
    # Note: get_gradcam returns an overlay RGB image. We need the raw mask or just compare heatmaps.
    # The get_gradcam function encapsulates mask generation.
    # We strip the mask returned now.
    orig_overlay, _ = get_gradcam(original_model, input_tensor, target_layer)
    orig_gray = cv2.cvtColor(orig_overlay, cv2.COLOR_RGB2GRAY)
    
    # 2. Randomize Model
    # We copy the model to avoid destroying the running model
    perturbed_model = copy.deepcopy(original_model)
    
    # Randomize the last convolutional layer (layer4) and the fc layer
    # ResNet50 structure: layer4 is the last block.
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
                
    perturbed_model.model.layer4.apply(init_weights)
    perturbed_model.model.fc.apply(init_weights)
    
    perturbed_model.to(next(original_model.parameters()).device)
    perturbed_model.eval()
    
    # 3. Get Perturbed Grad-CAM
    # Need to find the equivalent target layer in the new model instance
    if target_layer is None:
        new_target = perturbed_model.model.layer4[-1]
    else:
        # If user passed specific layer object, it belongs to original_model. 
        # We can't easily map it to perturbed_model without knowing the path.
        # Fallback to default behavior for sanity check if generic.
         new_target = perturbed_model.model.layer4[-1]
         
         new_target = perturbed_model.model.layer4[-1]
         
    perturbed_overlay, _ = get_gradcam(perturbed_model, input_tensor, new_target)
    perturbed_gray = cv2.cvtColor(perturbed_overlay, cv2.COLOR_RGB2GRAY)
    
    # 4. Compute SSIM
    score, _ = ssim(orig_gray, perturbed_gray, full=True)
    
    return score, perturbed_overlay

# --- LIME ---
def get_lime_explanation(model, input_tensor):
    """
    Generates LIME explanation superpixels.
    """
    model.eval()
    
    def batch_predict(images):
        """
        Input: List of numpy images (H, W, 3) normalized?
        LIME passes numpy arrays (0-255) if we start with that, or floats (0-1).
        We need to normalize them before passing to key model.
        """
        # Images come in as H,W,3 numpy arrays
        batch = torch.stack([
            torch.tensor(i.transpose(2, 0, 1), dtype=torch.float32) 
            for i in images
        ])
        
        # Normalize
        # LIME might pass images in range [0, 1] if input was [0, 1]
        # Our model expects normalized with mean/std
        # Assuming input images to this function are already standard 0-1 or 0-255?
        
        # The input_tensor provided to get_lime_explanation is ALREADY normalized.
        # But Lime generator works on an image representation.
        # Better Strategy: Pass denormalized image to LIME, and renorm inside predic func.
        
        # Actually, let's look at how LIME generates perturbations. It perturbs the `image` passed to `explain_instance`.
        # So `batch_predict` receives perturbed versions of that image.
        
        # Renormalize inputs
        # batch is N, 3, H, W
        # Standardize using same method
        mean = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
        
        batch = (batch - mean) / std
        
        device = next(model.parameters()).device
        batch = batch.to(device)
        
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)
            
        # LIME expects (N, num_classes)
        # We iterate binary probs to [prob_0, prob_1]
        probs = probs.cpu().numpy()
        return np.hstack([1-probs, probs])

    explainer = lime_image.LimeImageExplainer()
    
    # We need to pass a "viewable" image to LIME (vals 0-1)
    # So first denormalize to 0-1 float
    img_np = input_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = STD * img_np + MEAN
    img_np = np.clip(img_np, 0, 1) # This is what LIME sees
    
    print("Running LIME (this might take a moment)...")
    explanation = explainer.explain_instance(
        img_np.astype('double'), 
        batch_predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    # overlay boundaries on the image
    img_boundry = mark_boundaries(temp, mask)
    
    return (img_boundry * 255).astype(np.uint8), mask


# --- SHAP ---
def get_shap_explanation(model, input_tensor, background_batch):
    """
    Generates SHAP plots.
    input_tensor: (3, H, W)
    background_batch: (B, 3, H, W) - used for baseline
    """
    model.eval()
    
    # SHAP DeepExplainer or GradientExplainer
    # GradientExplainer is usually better for CNNs if layers are supported
    
    # We need to ensure batch dim
    to_explain = input_tensor.unsqueeze(0).to(next(model.parameters()).device)
    background = background_batch.to(next(model.parameters()).device)
    
    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values(to_explain)
    
    # shap_values is a list for each output class? 
    # For binary with single output, it might be a single tensor or list of 1.
    if isinstance(shap_values, list):
         print(f"SHAP values is list of length {len(shap_values)}")
         shap_values = shap_values[0]
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    
    # shap_values shape might be (1, 3, 224, 224, 1) for binary single output
    shap_numpy = np.array(shap_values)
    if shap_numpy.ndim == 5:
        shap_numpy = shap_numpy.squeeze(-1)
        
    shap_numpy = shap_numpy.transpose(0, 2, 3, 1)
    image_numpy = to_explain.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Denormalize image for display
    image_numpy = STD * image_numpy + MEAN
    
    return shap_numpy, image_numpy
