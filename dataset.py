import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader



def read_spectrogram(spectrogram_path):
    spectrogram = np.load(spectrogram_path)
    # Convert to torch float tensor and add channel dimension
    return torch.FloatTensor(spectrogram).unsqueeze(0)

def read_mood_vector(mood_vector_path):
    mood = np.load(mood_vector_path)
    # Convert to torch float tensor
    return torch.FloatTensor(mood)


class MoodyDataset(Dataset):
    """
        MoodyDataset is a PyTorch Dataset class for loading MoodySound dataset.
        The dataset is loaded from a CSV file containing metadata about the songs.
        The CSV file should contain the following columns:
        - spectrogram_file: path to the spectrogram file
        - target_file: path to the mood vector file
        Furthermore, the dataset is expected to be stored in the following directory structure:
        - data_dir/spectrograms/
        - data_dir/targets/
    """
    
    def __init__(self, config, transform=None, data_dir="/Volumes/Drive/MoodySound/data"):
        self.df = pd.read_csv(config)
        self.transform = transform
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() # to get the row of a csv
        

        spec_path = self.data_dir + '/spectrograms/' + self.df.iloc[idx]["spectrogram_file"]
        mood_path = self.data_dir + '/targets/' + self.df.iloc[idx]["target_file"]

        spec = read_spectrogram(spec_path)
        mood = read_mood_vector(mood_path)
        

        if self.transform:
            spec = self.transform(spec)
            mood = self.transform(mood)
        
        return spec, mood



# Do i need any transformations for the spectrogram or mood_vector?
# Remmember the mood_vector is already normalized


#dataloader = DataLoader(MoodyDataset(config="/Volumes/Drive/MoodySound/data/metadata.csv"))

#for spectrogram, target in dataloader:
#    print(f'Spec: {spectrogram}')
#    print(f'Target: {target}')