import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from src.model import PneumoniaNet
from src.config import BEST_MODEL_PATH, TRAIN_DIR
from src.explainers import get_gradcam, get_lime_explanation, get_shap_explanation, denormalize_image, sanity_check_gradcam, calculate_iou_attention
from src.dataset import PneumoniaDataset, get_transforms
from src.uncertainty import predict_with_uncertainty
from src.segmentation import LungSegmenter

# Set page config
st.set_page_config(
    page_title="PneumoXAI",
    page_icon="🫁",
    layout="wide"
)

# Constants
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# --- Helper Functions ---
@st.cache_resource
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PneumoniaNet()
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print("Loaded best model.")
    except FileNotFoundError:
        st.error(f"Model file not found at {BEST_MODEL_PATH}. Please run training first.")
        return None, device
    
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def get_background_batch(batch_size=5):
    """Load a small batch of training images to serve as background for SHAP."""
    # We need a transform that creates tensors
    transform = get_transforms('test') # Use test transform to just resize/norm
    dataset = PneumoniaDataset(
        root_dir=TRAIN_DIR,
        transform=transform
    )
    # Get a few images
    images = []
    # Just grab indices 0 to batch_size
    # Check bounds
    count = min(len(dataset), batch_size)
    for i in range(count):
        img, _ = dataset[i]
        images.append(img)
    
    if not images:
        return None
        
    if not images:
        return None
        
    return torch.stack(images)

def preprocess_image(uploaded_file, apply_mask=False):
    """
    Read file, optionally apply seg mask, then resize/norm.
    """
    # Read PIL
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    # 1. Apply Segmentation if requested BEFORE other processing?
    # Segmenter expects RGB 0-255.
    
    if apply_mask:
        segmenter = LungSegmenter()
        masked_img, mask = segmenter.segment(image_np)
        
        # Visualize the masking process in UI?
        # We'll return it so we can show it.
        image_np = masked_img # Continue with masked image

    # 2. Standard pipeline
    # Convert PIL RGB -> BGR (to simulate cv2 read) -> Gray
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_clahe = clahe.apply(gray)
    
    # Convert back to RGB
    image_rgb = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2RGB)
    
    # Albumentations
    transform = get_transforms('test') # Resize, Normalize, ToTensor
    augmented = transform(image=image_rgb)
    input_tensor = augmented['image']
    
    return image_np, input_tensor

# --- UI Layout ---

st.title("🫁 PneumoXAI: Trustworthy Medical Diagnostics")
st.markdown("""
This application uses **Explainable AI (XAI)** to assist identifying Pneumonia in Chest X-Rays.
It employs a **ResNet50** model fine-tuned on the dataset, and provides visual explanations using **Grad-CAM**, **LIME**, and **SHAP**.
""")

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.info("Model: ResNet50 (Binary Classification)")
use_lung_mask = st.sidebar.checkbox("Apply Lung Mask (Segmentation)")

# Load Resources
model, device = load_model()
background_batch = get_background_batch()

if background_batch is not None:
    background_batch = background_batch.to(device)

# File Uploader
uploaded_file = st.file_uploader("Upload a Chest X-Ray (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)
    
    # Preprocess
    preprocessed_img_numpy, input_tensor = preprocess_image(uploaded_file, apply_mask=use_lung_mask)
    input_tensor = input_tensor.to(device)
    
    # Debug Preprocessing
    show_preprocessed = st.checkbox("Show Preprocessed Image (CLAHE + Resize)")
    if show_preprocessed:
        st.image(preprocessed_img_numpy, caption="Preprocessed Input", width=300)
        if use_lung_mask:
            st.info("Lung mask applied.")

    # Inference
    mean_prob, uncertainty = predict_with_uncertainty(model, input_tensor, n_passes=20)
    prob = mean_prob
        
    # Prediction logic
    # dataset.py: class 1 is PNEUMONIA (usually, let's verify)
    # Yes, typically sorted alphabetically: NORMAL=0, PNEUMONIA=1
    # Check dataset.py loop: enumerate(['NORMAL', 'PNEUMONIA']) -> 0=NORMAL, 1=PNEUMONIA
    
    prediction_class = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = prob if prob > 0.5 else 1 - prob
    
    color = "red" if prediction_class == "PNEUMONIA" else "green"
    
    with col2:
        st.subheader("Prediction Result")
        st.markdown(f"<h2 style='color: {color};'>{prediction_class}</h2>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        st.markdown(f"**Uncertainty (Std Dev):** {uncertainty:.4f}")
        
        if uncertainty > 0.15:
            st.warning("⚠️ High Uncertainty - Clinical Review Recommended")
            
        st.progress(int(confidence * 100))

    st.markdown("---")
    st.header("🔍 Explainable AI Gallery")
    st.markdown("Visualizing model decision making process.")

    xai_col1, xai_col2, xai_col3 = st.columns(3)
    
    # Grad-CAM and LIME variables for Trust Score
    gradcam_mask = None
    lime_mask = None

    # Grad-CAM
    with xai_col1:
        st.subheader("Grad-CAM")
        try:
            with st.spinner("Generating Grad-CAM..."):
                gradcam_img, gradcam_mask = get_gradcam(model, input_tensor)
                st.image(gradcam_img, caption="Heatmap Focus", use_container_width=True)
        except Exception as e:
            st.error(f"Grad-CAM Error: {e}")

        # Sanity Check Button
        if st.button("Run Sanity Check (Randomization Test)"):
            st.write("Running Randomization Test...")
            ssim_score, perturbed_cam = sanity_check_gradcam(model, input_tensor)
            
            st.subheader("Sanity Check Results")
            st.write(f"SSIM Score: {ssim_score:.4f}")
            if ssim_score < 0.5:
                st.success("✅ Passed! heatmap changed significantly (Reference < 0.5).")
            else:
                st.error("❌ Failed! heatmap looks too similar to random model (Reference > 0.5).")
            
            # perturbed_cam is now (overlay, mask) tuple, need to unpack or fix src
            # Actually sanity_check_gradcam returns (score, perturbed_overlay_img)
            # as I didn't change its return signature in last step, only the get_gradcam call inside it.
            # wait, sanity_check_gradcam returns (score, perturbed_overlay). 
            # I updated sanity_check_gradcam to unpack get_gradcam correctly inside it.
            # So it returns image.
            st.image(perturbed_cam, caption="Perturbed Model Heatmap", use_container_width=True)

    # LIME
    with xai_col2:
        st.subheader("LIME Analysis")
        try:
            with st.spinner("Generating LIME (this takes a moment)..."):
                # LIME explainer inside explainers.py handles denormalization logic for its internal usage
                # But it expects normalized tensor as input
                lime_img, lime_mask = get_lime_explanation(model, input_tensor)
                st.image(lime_img, caption="Key Superpixels", use_container_width=True)
        except Exception as e:
            st.error(f"LIME Error: {e}")

    # SHAP
    with xai_col3:
        st.subheader("SHAP Features")
        try:
            with st.spinner("Generating SHAP..."):
                if background_batch is None:
                    st.warning("Background batch not available for SHAP.")
                else:
                    shap_vals, shap_img = get_shap_explanation(model, input_tensor, background_batch)
                    
                    # shap.image_plot returns a matplotlib figure usually or plots directly
                    # Our helper returns numpy arrays. We need to plot them.
                    
                    # We can use shap.image_plot directly in app potentially?
                    # Or plotting manually.
                    # Let's use matplotlib to render the shap values 
                    
                    # Actually shap.image_plot(shap_values, pixel_values) plots to the current figure if show=False
                    import shap
                    plt.figure() # Create new figure
                    shap.image_plot(shap_vals, shap_img, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()
                    
        except Exception as e:
            st.error(f"SHAP Error: {e}")

    st.markdown("---")
    st.header("🤝 Explanation Agreement (Trust Score)")
    if gradcam_mask is not None and lime_mask is not None:
        iou = calculate_iou_attention(gradcam_mask, lime_mask)
        col_trust1, col_trust2 = st.columns([1, 3])
        with col_trust1:
             st.metric("IoU Agreement", f"{iou:.2f}")
        with col_trust2:
             if iou > 0.3: # Threshold for "good enough" agreement
                 st.success(f"**High Agreement (Trustworthy)**: Grad-CAM and LIME focus on similar regions.")
                 st.progress(min(1.0, iou * 2)) # Scale for visuals
             else:
                 st.error(f"**Low Agreement (Caution)**: Grad-CAM and LIME highlight different regions. Interpret with care.")
                 st.progress(min(1.0, iou * 2))
    else:
        st.info("Run both Grad-CAM and LIME to see the Agreement Score.")

