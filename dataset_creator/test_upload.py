from generate_data_stats import get_dataset_statistics_from_comprehensive_mood
import pandas as pd
import boto3
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

load_dotenv()

s3 = boto3.client('s3')

s3.download_file(
    Bucket=os.getenv('S3_BUCKET_NAME'),
    Key='data/shuffled_metadata.csv',
    Filename='dataset_creator/shuffled_metadata.csv'
)

def search_bucket_for_missing_file(row):
    try:
        s3.head_object(
            Bucket=os.getenv('S3_BUCKET_NAME'),
            Key='data/spectrograms/' + row.spectrogram_file
        )

        s3.head_object(
            Bucket=os.getenv('S3_BUCKET_NAME'),
            Key='data/targets/' + row.target_file
        )

        print(f"Song found: {row.title}")
        return None
    
    except Exception as e:
        print(f"Song not found: {row.title}")
        return row.title

def search_for_missing_files(csv_path):

    df = pd.read_csv(csv_path)

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(filter(None, executor.map(search_bucket_for_missing_file, df.itertuples())))

    return results



def save_missing_files(missing_files, output_path):
    with open(output_path, 'w') as f:
        for file_name in missing_files:
            f.write(f"{file_name}\n")
    
    print(f"Results saved to {output_path}")



def main():
    missing_files = search_for_missing_files('dataset_creator/shuffled_metadata.csv')
    save_missing_files(missing_files, 'dataset_creator/missing_files.txt')






if __name__ == "__main__":
    main()



