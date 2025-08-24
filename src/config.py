from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import torch

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "dataset_1"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CSV_DATA_DIR = DATA_DIR / "csv"
FULL_SIZE_RAW_MAPS_DIR = DATA_DIR / "full_size_maps"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS_DIR = PROJ_ROOT / "model_weights"

BATCH_SIZE_TRANSFORMER = 32
BATCH_SIZE_RESNET = 128

MAX_EPOCHS = 25

torch.set_float32_matmul_precision('high')

