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

    # Normalize the vector
    norm = np.linalg.norm(mood)
    normalized_mood = mood / norm

    # Convert to torch float tensor
    return torch.FloatTensor(normalized_mood).squeeze(0)

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
    
    def __init__(self, config, transform=None, cache_dir='/data/cache'):
        self.df = pd.read_csv(config)
        self.transform = transform
        self.cache_dir = cache_dir
        
        # Verify cache directory exists
        if not os.path.exists(cache_dir):
            raise ValueError(f"Cache directory {cache_dir} does not exist. Please run cache_dataset.py first.")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Use cache directory to load spectrogram and mood vector
        spec_path = os.path.join(self.cache_dir, 'spectrograms', self.df.iloc[idx]["spectrogram_file"])
        mood_path = os.path.join(self.cache_dir, 'targets', self.df.iloc[idx]["target_file"])
        
        # Verify files exist in cache
        if not os.path.exists(spec_path) or not os.path.exists(mood_path):
            raise FileNotFoundError(f"Files not found in cache directory. Spectrogram: {spec_path}, Mood: {mood_path}")

        spec = read_spectrogram(spec_path)
        mood = read_mood_vector(mood_path)
        
        if self.transform:
            spec = self.transform(spec)
        
        return spec, mood






def test():
    

    dataset = MoodyDataset(config="/data/cache/data/metadata.csv", 
                           parent_dir_path="/Volumes/Drive/MoodySound/data")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(next(iter(dataloader)))
    print(next(iter(dataloader)))
    print(next(iter(dataloader)))


if __name__ == "__main__":
    test()

