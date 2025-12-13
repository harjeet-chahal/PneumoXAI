# PneumoXAI: Trustworthy Medical Diagnostics 🫁

**A Trustworthy AI system for Pneumonia detection using ResNet50 and Comparative XAI.**

## Project Overview
PneumoXAI is a computer vision project designed to assist medical professionals in identifying pneumonia from chest X-rays. Unlike "black-box" models, this system prioritizes interpretability by integrating multiple Explainable AI (XAI) techniques. It provides visual evidence for its predictions, fostering trust and collaboration between AI and clinicians.

The core model is a Fine-Tuned **ResNet50** optimized for binary classification (Normal vs. Pneumonia), featuring a custom data pipeline with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for enhanced feature visibility.

## Tech Stack
*   **Deep Learning**: PyTorch, Torchvision (ResNet50)
*   **Image Processing**: OpenCV (CLAHE), Albumentations (Augmentation)
*   **Explainable AI (XAI)**:
    *   **Grad-CAM**: Gradient-weighted Class Activation Mapping for heatmap visualization.
    *   **LIME**: Local Interpretable Model-agnostic Explanations for superpixel analysis.
    *   **SHAP**: SHapley Additive exPlanations for feature importance.
*   **Frontend**: Streamlit
*   **Data Handling**: NumPy, Pandas

## How to Run

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
To train the ResNet50 model:
```bash
python src/train.py
```
*   The trained model will be saved to `models/best_model.pth`.
*   Training logs (Accuracy, Recall, F1) are displayed in the console.

### 3. Running the App
Launch the interactive web application:
```bash
streamlit run app.py
```
*   Upload a Chest X-Ray image (JPG/PNG).
*   View the prediction (Normal/Pneumonia) and Confidence Score.
*   Explore the XAI Gallery to seeing Grad-CAM, LIME, and SHAP visualizations.

## Results
The model was trained for 8 epochs with Early Stopping. The best performing model (Epoch 4) achieved:

* **Recall: 100% (1.00)** - *Critical for medical diagnosis to ensure zero False Negatives.*
* **F1-Score: 94.1%** - *Indicates a strong balance between precision and recall.*
* **Accuracy: 93.75%** - *High overall correctness on unseen data.*

## Directory Structure
```
PneumoXAI/
├── data/               # Raw dataset
├── models/             # Saved model checkpoints
├── src/
│   ├── config.py       # Configuration and paths
│   ├── dataset.py      # Custom Dataset and DataLoaders
│   ├── explainers.py   # XAI implementations (Grad-CAM, LIME, SHAP)
│   ├── model.py        # PneumoniaNet architecture
│   └── train.py        # Training loop
├── app.py              # Streamlit Frontend
├── requirements.txt    # Project dependencies
└── README.md           # Project Documentation
```
