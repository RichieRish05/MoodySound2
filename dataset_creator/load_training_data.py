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

    # Calculate the L2 norm (Euclidean norm) of the vector
    norm = np.linalg.norm(mood_vector)

    # Normalize mood vector
    normalized_vector = mood_vector / norm
    
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






def load_training_vectors(csv_path: Path, output_directory = None, start_index=0, num_rows=1000):
    # Create a pandas data frame to iterate through the csv
    df = pd.read_csv(csv_path, skiprows=range(1, start_index + 1), nrows=num_rows)
    print(f"Processing rows {start_index} to {start_index + len(df)}")

    # Create a list to store metadata
    metadata = []

    # Iterate through the csv
    for _, row in df.iterrows():
        spectrograms = build_spectogram_data(row)
        
        if not spectrograms:
            continue
       
        mood_vector = build_mood_vector(row)
        cleaned_name = sanitize_song_name(row["title"])
        
        for index, spectrogram in enumerate(spectrograms):
            # Create file names
            spectrogram_file = f'{cleaned_name}_{(index+1)*10}s_matrix.npy'
            target_file = f'{cleaned_name}_{(index+1)*10}s_target.npy'
            
            # Save the data and associated target to .npy files
            matrix_file_path = f'{output_directory}/spectograms/{spectrogram_file}'
            np.save(matrix_file_path, spectrogram)
            print(f'Saved matrix to {matrix_file_path}')
            
            target_file_path = f'{output_directory}/targets/{target_file}'
            np.save(target_file_path, mood_vector)
            print(f'Saved mood vector to {target_file_path}')
            
            # Add to metadata
            metadata.append({
                'spectrogram_file': spectrogram_file,
                'target_file': target_file,
                'title': row["title"],
                'artist': row["artist"]
            })

        # Introduce a delay to prevent overloading the API
        time.sleep(1)  # Adjust the delay time as needed

    # Create metadata DataFrame
    new_metadata_df = pd.DataFrame(metadata)
    
    # Check if metadata file exists
    metadata_path = f'{output_directory}/metadata.csv'
    if Path(metadata_path).exists():
        # Read existing metadata and append new data
        existing_metadata = pd.read_csv(metadata_path)
        combined_metadata = pd.concat([existing_metadata, new_metadata_df], ignore_index=True)
        combined_metadata.to_csv(metadata_path, index=False)
    else:
        # If no existing metadata, save new metadata
        new_metadata_df.to_csv(metadata_path, index=False)



drive_name = "/Volumes/Drive/MoodySound/data"

load_training_vectors(
    csv_path=Path("/Users/rishi/MoodySound/dataset_creator/augmented.csv"), 
    output_directory=Path(drive_name),
    start_index=0,
    num_rows=10
)
