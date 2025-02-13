import pandas as pd
import shutil
import os

def create_a_subset_of_the_dataset(config='/Volumes/Drive/MoodySound/data/shuffled_metadata.csv', output_path='/Volumes/Drive/MoodySound/test_data'):
    parent_dir = os.path.dirname(config)

    # Create the output directory structure if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + '/spectrograms', exist_ok=True)
    os.makedirs(output_path + '/targets', exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(config, nrows=1000)


    for index, row in df.iterrows():

        spec_path = os.path.join(parent_dir + '/spectrograms', row['spectrogram_file'])
        target_path = os.path.join(parent_dir + '/targets', row['target_file'])

        print(spec_path, target_path)

        # Copy the spectrogram file
        shutil.copy(spec_path, output_path + '/spectrograms')

        # Copy the target file
        shutil.copy(target_path, output_path + '/targets')

    df.to_csv(output_path + '/shuffled_metadata.csv', index=False)


if __name__ == "__main__":
    create_a_subset_of_the_dataset()
 