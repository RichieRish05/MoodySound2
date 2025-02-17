from generate_data_stats import get_dataset_statistics_from_comprehensive_mood
import pandas as pd
import boto3
import os
from dotenv import load_dotenv
from pathlib import Path
import unicodedata




def download_file(bucket_name, key, file_path):
    s3 = boto3.client('s3')

    s3.download_file(
        Bucket=bucket_name,
        Key=key,
        Filename=file_path
    )


def normalize_filename(filename):
    # Normalize to NFKD form and encode as ASCII to handle special characters
    return unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')


def filter_name(file):
    file_name = file['Key']
    if 'spectrograms' in file_name or 'targets' in file_name:
        return normalize_filename(Path(file_name).name)
    return None


def get_all_objects_in_bucket(bucket_name, prefix = 'data/'):
    all_objects = set()

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket = bucket_name, Delimiter = '/', Prefix = prefix)

    for page in page_iterator:
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                all_objects.update(get_all_objects_in_bucket(bucket_name, prefix = prefix['Prefix']))
        
        if 'Contents' in page and page['Contents']:
            for file in page['Contents']:
                if name := filter_name(file):
                    all_objects.add(name)
    
    return all_objects


def get_all_spec_and_targets_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Normalize filenames from CSV using the normalization function
    spec_files = {normalize_filename(name) for name in df['spectrogram_file']}
    target_files = {normalize_filename(name) for name in df['target_file']}
    return spec_files | target_files

def update_csv(csv_path, missing_files, bucket):
    df = pd.read_csv(csv_path)

    mask = ~(df['spectrogram_file'].apply(normalize_filename).isin(missing_files) | 
             df['target_file'].apply(normalize_filename).isin(missing_files))
    df = df[mask]
    df.to_csv(csv_path, index=False)

    s3 = boto3.client('s3')

    s3.upload_file(
        Bucket = bucket,
        Key = 'shuffled_metadata.csv',
        Filename = csv_path
    )

    return df


def main():
    load_dotenv()
    bucket = os.getenv('S3_BUCKET_NAME')


    uploaded_objects = get_all_objects_in_bucket(bucket)
    total_objects = get_all_spec_and_targets_from_csv('dataset_creator/shuffled_metadata.csv')
    missing_objects = total_objects - uploaded_objects
    print(len(uploaded_objects))
    print(len(total_objects))
    print(len(missing_objects))
    #update_csv('dataset_creator/shuffled_metadata.csv', missing_objects, bucket)

    

    return missing_objects

if __name__ == '__main__':
    load_dotenv()
    main()