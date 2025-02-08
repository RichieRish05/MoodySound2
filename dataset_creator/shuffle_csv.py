import pandas as pd
import random

def shuffle_csv_files(input_csv_path, output_csv_path):
    
    # Read the csv file
    df = pd.read_csv(input_csv_path).copy()


    
    # Shuffle the Data Frame
    shuffled_df = df.sample(frac=1, random_state=random.randint(1, 1000)).reset_index(drop=True)
    
    # Save to new CSV
    shuffled_df.to_csv(output_csv_path, index=False)
    
    print(f"Shuffled CSV saved as: {output_csv_path}")


if __name__ == "__main__":
    shuffle_csv_files('/Volumes/Drive/MoodySound/data/metadata.csv', "/Volumes/Drive/MoodySound/data/shuffled_metadata.csv")