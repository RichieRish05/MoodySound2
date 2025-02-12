import boto3
import os
import time
from dotenv import load_dotenv
import boto3
import pandas as pd

load_dotenv()
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
CACHE_DIR = './test'
MAX_RETRIES = 3

def download_to_VASTAI_instance_with_retry():
    # Use AWS CLI for efficient download
    s3 = boto3.client('s3')


def download_dir(s3, dist = 'data/', bucket=BUCKET_NAME):
    os.makedirs(CACHE_DIR, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix = dist)



    for page in page_iterator:
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                download_dir(s3,prefix['Prefix'], bucket)
                
        if 'Contents' in page:
            for file in page['Contents']:
                dest_pathname = os.path.join(CACHE_DIR, file['Key'])

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(dest_pathname), exist_ok=True)
                
                # Download the file
                print(f"Downloading {file['Key']} to {dest_pathname}")
                try:
                    s3.download_file(
                        Bucket=bucket,
                        Key=file['Key'],
                        Filename=dest_pathname
                    )
                except Exception as e:
                    print(f"Error downloading {file['Key']}: {e}")
            



def main():
    s3 = boto3.client('s3')
    download_dir(s3)


main()
"""
Create all accessible directories before running these commands

aws configure
aws s3 sync s3://rishitestbucket01/data/ /mnt/data

"""


"""
sudo mkfs -t ext4 /dev/

"""