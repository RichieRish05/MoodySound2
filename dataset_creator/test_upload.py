from generate_data_stats import get_dataset_statistics_from_comprehensive_mood
import pandas as pd
import boto3
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unicodedata

load_dotenv()

s3 = boto3.client('s3')


# s3.download_file(
#     Bucket= os.getenv('S3_BUCKET_NAME'),
#     Key='data/shuffled_metadata.csv',
#     Filename='dataset_creator/shuffled_metadata.csv'
# )


def normalize_filename(filename):
    # Normalize to NFKD form and encode as ASCII to handle special characters
    return unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')


def filter_paths(file):
    file_name = file['Key']
    if 'spectrograms' in file_name or 'targets' in file_name:
        return normalize_filename(Path(file_name).name)
    return None


def get_all_objects_in_bucket(bucket_name, prefix = 'data/'):
    all_objects = set()

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket = bucket_name, Delimiter = '/', Prefix = prefix)

    for page in page_iterator:
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                all_objects.update(get_all_objects_in_bucket(bucket_name, prefix = prefix['Prefix']))
        
        if 'Contents' in page and page['Contents']:
            for file in page['Contents']:
                if name := filter_paths(file):
                    all_objects.add(name)
    
    return all_objects


def get_all_spec_and_targets_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Normalize filenames from CSV using the same normalization function
    spec_files = {normalize_filename(name) for name in df['spectrogram_file']}
    target_files = {normalize_filename(name) for name in df['target_file']}
    return spec_files | target_files

def update_csv(csv_path, missing_files):
    df = pd.read_csv(csv_path)

    mask = ~(df['spectrogram_file'].apply(normalize_filename).isin(missing_files) | df['target_file'].apply(normalize_filename).isin(missing_files))
    df = df[mask]
    df.to_csv(csv_path, index=False)


def main():
    all_objects = get_all_objects_in_bucket('moody')
    all_intended_objects = get_all_spec_and_targets_from_csv('dataset_creator/shuffled_metadata.csv')


    missing_files = all_intended_objects - all_objects
    update_csv('dataset_creator/shuffled_metadata.csv', missing_files)

    print(len(missing_files))



main()