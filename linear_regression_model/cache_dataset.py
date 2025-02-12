import boto3
import os
import time
from dotenv import load_dotenv
import boto3

load_dotenv()
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
CACHE_DIR = './test'
MAX_RETRIES = 3

def download_to_VASTAI_instance_with_retry():
    # Use AWS CLI for efficient download
    s3 = boto3.client('s3')


def download_dir(dist='data/', bucket=BUCKET_NAME):
    os.makedirs(CACHE_DIR, exist_ok=True)
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix = dist)



    for page in page_iterator:
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                download_dir(prefix['Prefix'], bucket)
                
        if 'Contents' in page:
            for file in page['Contents']:
                 dest_pathname = os.path.join(CACHE_DIR, file['Key'])

                # Create directory if it doesn't exist
                 os.makedirs(os.path.dirname(dest_pathname), exist_ok=True)
                
                # Download the file
                 print(f"Downloading {file['Key']} to {dest_pathname}")
                 s3.download_file(
                     Bucket=bucket,
                     Key=file['Key'],
                     Filename=dest_pathname
                 )



            
    



download_dir()


"""
Create all accessible directories before running these commands

aws configure
aws s3 sync s3://rishitestbucket01/data/ /mnt/data

"""


"""
sudo mkfs -t ext4 /dev/

"""