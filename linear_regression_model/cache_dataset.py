import boto3
import os
import time
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
CACHE_DIR = '/cached-data'
MAX_RETRIES = 3

def download_to_EC2_instance_with_retry():
    s3 = boto3.client('s3')
    
    print("Starting download process...")
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    for try_num in range(MAX_RETRIES):
        try:
            # Use AWS CLI for efficient download
            os.system(f'aws s3 sync s3://{BUCKET_NAME} {CACHE_DIR} --progress')


            # Verify download is complete
            if os.path.exists(CACHE_DIR) and len(os.listdir(CACHE_DIR)) > 0:
                print("Download complete!")
                os.system(f'ls -l {CACHE_DIR}')
                os.system(f'du -sh {CACHE_DIR}')
                break
            else:
                # If download is not complete, retry
                raise Exception("Download failed")
            
        except Exception as e:
            print(f"Error: {e}")
            print(f"Download failed on attempt {try_num+1}. Retrying...")
            time.sleep(10)


"""
Create all accessible directories before running these commands

aws configure
aws s3 sync s3://rishitestbucket01/data/ /cached-data

"""


"""
sudo mkfs -t ext4 /dev/nvme1n1


"""