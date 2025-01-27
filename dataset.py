import pandas as pd
import numpy as np
import gcsfs
import torch
from torch.utils.data import Dataset, DataLoader



def read_spectrogram(spectrogram_path, fs):
    with fs.open(spectrogram_path, 'rb') as f:
        spectrogram = np.load(f)
    # Convert to torch float tensor and add channel dimension
    return torch.FloatTensor(spectrogram).unsqueeze(0)

def read_mood_vector(mood_vector_path, fs):
    with fs.open(mood_vector_path, 'rb') as f:
        mood = np.load(f)
    # Convert to torch float tensor
    return torch.FloatTensor(mood).squeeze(0)

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
    
    def __init__(self, config, file_system, transform=None):
        with file_system.open(config, 'rb') as f:
            self.df = pd.read_csv(f)
        self.transform = transform
        self.file_system = file_system
         # Extract bucket path from config
        self.bucket_path = '/'.join(config.split('/')[:-1])  # removes metadata.csv from path
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() # to get the row of a csv
        

        spec_path = f'{self.bucket_path}/spectrograms/' + self.df.iloc[idx]["spectrogram_file"]
        mood_path = f'{self.bucket_path}/targets/' + self.df.iloc[idx]["target_file"]

        spec = read_spectrogram(spec_path, self.file_system)
        mood = read_mood_vector(mood_path, self.file_system)
        

        if self.transform:
            spec = self.transform(spec)
        
        return spec, mood






def test():
    fs = gcsfs.GCSFileSystem(project='testmoodysound',
                            token='/Users/rishi/Desktop/Google Cloud Keys/testmoodysound-e7d906478321.json')
    

    dataset = MoodyDataset(config="gs://moodysoundtestbucket/metadata.csv", file_system=fs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(next(iter(dataloader)))
    print(next(iter(dataloader)))


if __name__ == "__main__":
    test()

