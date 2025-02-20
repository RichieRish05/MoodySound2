import numpy as np
import torch
import sys
import os

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the model
from model_production.custom_model import MoodyConvNet


def load_model(path):
    model = MoodyConvNet()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def model_pass(model, spectrogram):
    model.eval()
    with torch.no_grad():
        output = model(spectrogram)
        return output
    

def main():
    model = load_model('model_loader/best_model.pth')

    
if __name__ == '__main__':
    main()

