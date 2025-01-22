import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


config = "/Volumes/Drive/MoodySound/data/metadata.csv"


def read_spectrogram(spectrogram_path):
    return np.load(spectrogram_path)

def read_mood_vector(mood_vector_path):
    return np.load(mood_vector_path)


class MoodyDataset(Dataset):
    
    def __init__(self, config, transform=None):
        self.df = pd.read_csv(config)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {
            "spectrogram": read_spectrogram(self.df.iloc[idx]["spectrogram_file"]),
            "mood_vector": read_mood_vector(self.df.iloc[idx]["target_file"])
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

dataset = MoodyDataset(config=config)
dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True, num_workers=2)



