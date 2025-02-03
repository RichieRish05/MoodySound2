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
    
    return normalized_vector


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






def load_training_vectors(csv_path: Path, output_directory = None, start_index=0, num_rows=None):
    try:
        # Create a pandas data frame to iterate through the csv
        df = pd.read_csv(csv_path, skiprows=range(1, start_index + 1), nrows=num_rows)
        total_rows = len(df)
        print(f"Processing rows {start_index} to {start_index + total_rows}")

        # Create a list to store metadata
        metadata = []
        
        # Track progress
        processed_count = 0
        error_count = 0

        # Iterate through the csv
        for _, row in df.iterrows():
            try:
                spectrograms = build_spectogram_data(row)
                
                if not spectrograms:
                    print(f"Skipping {row['title']} - No audio preview found")
                    error_count += 1
                    continue
                
                mood_vector = build_mood_vector(row)
                cleaned_name = sanitize_song_name(row["title"])
                
                for index, spectrogram in enumerate(spectrograms):
                    # Create file names
                    spectrogram_file = f'{cleaned_name}_{(index+1)*10}s_matrix.npy'
                    target_file = f'{cleaned_name}_{(index+1)*10}s_target.npy'
                    
                    # Save the data and associated target to .npy files
                    matrix_file_path = f'{output_directory}/spectrograms/{spectrogram_file}'
                    np.save(matrix_file_path, spectrogram)
                    
                    target_file_path = f'{output_directory}/targets/{target_file}'
                    np.save(target_file_path, mood_vector)
                    
                    # Add to metadata
                    metadata.append({
                        'spectrogram_file': spectrogram_file,
                        'target_file': target_file,
                        'title': row["title"],
                        'artist': row["artist"],
                        'comprehensive_mood': row['comprehensive_mood']
                    })

                processed_count += 1
                
                # Print progress every 1000 songs
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count}/{total_rows} songs. Errors: {error_count}")
                    
                    # Save metadata periodically
                    save_metadata(metadata, output_directory)
                    metadata = []  # Clear the metadata list after saving

                # Introduce a delay to prevent overloading the API
                time.sleep(1)

            except Exception as e:
                print(f"Error processing {row['title']}: {str(e)}")
                error_count += 1
                continue

        # Save any remaining metadata
        if metadata:
            save_metadata(metadata, output_directory)

        print(f"\nProcessing complete!")
        print(f"Successfully processed: {processed_count}")
        print(f"Errors encountered: {error_count}")

    except Exception as e:
        print(f"Fatal error occurred: {str(e)}")
        raise

def save_metadata(metadata, output_directory):
    """Helper function to save metadata to CSV"""
    if not metadata:
        return
        
    new_metadata_df = pd.DataFrame(metadata)
    metadata_path = f'{output_directory}/metadata.csv'
    
    if Path(metadata_path).exists():
        existing_metadata = pd.read_csv(metadata_path)
        combined_metadata = pd.concat([existing_metadata, new_metadata_df], ignore_index=True)
        # Drop duplicates based on spectrogram_file
        combined_metadata = combined_metadata.drop_duplicates(subset=['spectrogram_file'], keep='last')
        combined_metadata.to_csv(metadata_path, index=False)
    else:
        new_metadata_df.to_csv(metadata_path, index=False)

# Path on seagate external hardrive
drive_name = "/Volumes/Drive/MoodySound/data"

load_training_vectors(
    csv_path=Path("/Users/rishi/MoodySound2/dataset_creator/augmented.csv"), 
    output_directory=Path(drive_name),
    start_index=40000,  # Start from here
    num_rows=4000   # Process 10,000 rows at a time
)

"""
# Remove existing metadata files from your external drive in terminal after running this script to remove unwanted overhead
find /Volumes/Drive/MoodySound/data -name "._*" -delete
find /Volumes/Drive/MoodySound/data -name ".DS_Store" -delete
"""