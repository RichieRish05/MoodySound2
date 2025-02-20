import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os



def read_spectrogram(spectrogram_path):
    # Load in the spectrogram
    spectrogram = np.load(spectrogram_path)
    # Convert to torch float tensor and add channel dimension
    return torch.FloatTensor(spectrogram).unsqueeze(0)

def read_mood_vector(mood_vector_path):
    # Load in the mood vector
    mood = np.load(mood_vector_path)

    # Create a valid probability distribution
    mood = mood / np.sum(mood)

    # Convert to log space for KL divergence
    mood = np.log(mood + 1e-10)

    # Convert to torch float tensor
    return torch.FloatTensor(mood).unsqueeze(0)

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
    
    def __init__(self, config, transform=None):
        self.df = pd.read_csv(config)
        self.transform = transform
        self.data_dir = os.path.dirname(config)
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Use data directory to load spectrogram and mood vector
        spec_path = os.path.join(self.data_dir, 'spectrograms', self.df.iloc[idx]["spectrogram_file"])
        mood_path = os.path.join(self.data_dir, 'targets', self.df.iloc[idx]["target_file"])
        
        
        try:
            spec = read_spectrogram(spec_path)
            mood = read_mood_vector(mood_path)
                    
            if self.transform:
                spec = self.transform(spec)
            
            return spec, mood
        
        except Exception as e:
            return self.__getitem__((idx+1) % len(self))



# THINK ABOUT NORMALIZING THE SPECTROGRAM



def test():
    

    dataset = MoodyDataset(
        config="/Volumes/Drive/MoodySound/test_data/shuffled_metadata.csv"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(next(iter(dataloader)))
    s, t = next(iter(dataloader))



if __name__ == "__main__":
    test()

