import sys
import os

# Add 'src' to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import get_data
from model import SimpleModel

def train():
    data = get_data()
    model = SimpleModel()
    # Save checkpoint to /models
    model.save("models/checkpoint_01.pt")
    print("Training complete.")

if __name__ == "__main__":
    train()