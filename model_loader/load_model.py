import torch
from dotenv import load_dotenv
import os
from model_production import MoodyConvNet
import numpy as np

load_dotenv()

class MoodPredictor():
    def __init__(self, model_path, model_weights_path):
        self.model_path = model_path
        self.model_weights_path = model_weights_path
        self.model = None

    def load_model_info(self):
        return torch.load(self.model_weights_path, map_location=torch.device('cpu'))
    
    def save_model(self, model, path):
        torch.save(model, path)
    
    def create_model(self):
        model_info = self.load_model_info()
        model = MoodyConvNet()
        model.load_state_dict(model_info['model_state_dict'])
        self.save_model(model, 'model_loader/best_model.pth')
        self.model = model
        return model

    def load_model(self, path):
        self.model = torch.load(path, map_location=torch.device('cpu'))
        return self.model
    
    def predict(self, s):
        self.model.eval()
        with torch.no_grad():
            s = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)
            output = self.model(s)
            return output
        
    def get_target(self, target_path):
        target = np.load(target_path)
        norm = np.linalg.norm(target)
        target = target / norm
        return torch.FloatTensor(target).squeeze()


if __name__ == "__main__":
    mood_predictor = MoodPredictor('model_loader/best_model.pth', 'model_loader/best_model_weights.pth')
    mood_predictor.create_model()
    s = np.load('/Users/rishi/MoodySound2/test/data/spectrograms/Bohemian_Rhapsody_20s_matrix.npy')
    t = mood_predictor.get_target('/Users/rishi/MoodySound2/test/data/targets/Bohemian_Rhapsody_20s_target.npy')
    output = mood_predictor.predict(s)
    print(output)
    print(t)

    


        