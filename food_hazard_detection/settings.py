import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

PROJECT_DIR = Path(__file__).parent
FILES_DIR = PROJECT_DIR / "files"
SYNTHETIC_DATA_DIR = FILES_DIR / "datasets" / "synthetic_data"
