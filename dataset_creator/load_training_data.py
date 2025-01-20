from pathlib import Path
import pandas as pd
import numpy as np
import time
from song_preview import get_song_preview
from spectrogram import get_spectrogram_data




def sanitize_song_name(song_name):
    # Replace spaces with underscores
    song_name = song_name.strip().replace(' ', '_')

    invalid_characters = ['/', '\\',':', '*', '?', '"', '<', '>', '|' ]

    # Remove invalid characters
    sanitized_song_name = ''.join(char for char in song_name if char not in invalid_characters)


    return sanitized_song_name


def build_mood_vector(row):
    # All of the features in our mood vector
    mood_features = ["danceability",
                     "mood_acoustic",
                     "mood_aggressive", 
                     "mood_electronic", 
                     "mood_happy", 
                     "mood_party", 
                     "mood_relaxed",
                     "mood_sad"]
    
    # Search through the csv row for the mood vector
    mood_vector = []
    for feature in mood_features:
        mood_vector.append(row[feature])
    
    mood_vector = np.array([mood_vector])
    
    return mood_vector


def build_spectogram_data(row):
    """
    Parse a mood csv row and build a spectogram based on the 
    title and artist of the song
    """
    # Build the search query
    search_query = row["title"] + ' ' + row["artist"]

    # Make the deezer api call to get an audio file
    audio_url = get_song_preview(search_query)

    # Return None if the deezer api does not return an audio file
    if not audio_url:
        return None
           
        
    # Build a spectrogram with the audio file
    spectograms = get_spectrogram_data(audio_url)


    return spectograms






def load_training_vectors(csv_path: Path, output_directory = None):
    # Create a pandas data frame to iterate through the csv
    df = pd.read_csv(csv_path)


    # Iterate through the csv
    for _, row in df.iterrows():
        spectrograms = build_spectogram_data(row)
        
        if not spectrograms:
            continue
       
        mood_vector = build_mood_vector(row)


        cleaned_name = sanitize_song_name(row["title"])
        for index, spectrogram in enumerate(spectrograms):
            # Save the data and associated target to .npy files
            file_path = f'{output_directory}/{cleaned_name}_{(index+1)*10}s'
            np.save(f'{file_path}_matrix', spectrogram)
            print(f'Saved matrix to {file_path}_matrix')
            np.save(f'{file_path}_target', mood_vector)
            print(f'Saved mood vector to {file_path}_target')
 
        




        # Introduce a delay to prevent overloading the API
        time.sleep(1)  # Adjust the delay time as needed





load_training_vectors(Path("/Users/rishi/MoodySound/dataset_creator/mood.csv"), Path("/Users/rishi/MoodySound/dataset/train"))

#print(sanitize_song_name("Hard Way To Make An Easy Living The Bellamy Brothers"))