import os

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Model Save Path
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
