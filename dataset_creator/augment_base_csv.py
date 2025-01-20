import pandas as pd
from pathlib import Path

def augment_dataset(base_csv: Path, larger_csv: Path, number_of_songs_per_split: int, output_csv: Path) -> None:
    """
    Augment the base CSV file (original) with the additional data from the larger CSV file to ensure each mood 
    contains an equal amount of songs.
    """

    # Pandas DataFrames
    base_df = pd.read_csv(base_csv).round(20)
    larger_df = pd.read_csv(larger_csv).round(20)


    # Metadata headers
    metadata = ["title", "artist"]

    # Moods
    mood_features = [
        "danceability", "mood_acoustic", "mood_aggressive", "mood_electronic",
        "mood_happy", "mood_party", "mood_relaxed", "mood_sad"
    ]

    # Create an empty data frame that represents the new csv 
    augmented_df = pd.DataFrame(columns = metadata + mood_features)

    # Count the number of songs per mood in the base csv
    mood_counts = base_df[mood_features].idxmax(axis=1).value_counts().to_dict()

    # Augment each mood category as needed
    for mood in mood_features:
        songs_that_match_current_mood = mood_counts[mood]
        if songs_that_match_current_mood < number_of_songs_per_split:
            number_of_songs_to_add = number_of_songs_per_split - songs_that_match_current_mood
            print(f'Adding {number_of_songs_to_add} {mood} songs')
            
            # Get songs where this mood has the highest value among all moods
            songs_to_add = larger_df[larger_df[mood_features].idxmax(axis=1) == mood].copy()
            songs_to_add = songs_to_add.sample(n=number_of_songs_to_add, replace=False)
            songs_from_base = base_df[base_df[mood_features].idxmax(axis=1) == mood].copy()
            songs_to_add = pd.concat([songs_from_base, songs_to_add], ignore_index=True)
        else:
            print(f'Slicing data to {number_of_songs_per_split} {mood} songs')
            # Get songs from base_df where this mood has the highest value
            songs_to_add = base_df[base_df[mood_features].idxmax(axis=1) == mood].copy()
            songs_to_add = songs_to_add.sample(n=number_of_songs_per_split, replace=False)




        # Add to augmented_df
        if augmented_df.empty:
            augmented_df = songs_to_add
        else:
            augmented_df = pd.concat([augmented_df, songs_to_add], ignore_index=True)



    




    # Shuffle the augmented dataset
    # augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)

    



    # Save the augmented dataset to the specified output CSV
    augmented_df.to_csv(output_csv, index=False, float_format='%.20f')



# Example usage:
#augment_dataset(Path("misc/base.csv"), Path("misc/larger.csv"), 15, Path("dataset_creator/augmented.csv"))

