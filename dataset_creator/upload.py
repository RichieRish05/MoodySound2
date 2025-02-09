import os
from dotenv import load_dotenv
import boto3

load_dotenv()
data_path = '/Volumes/Drive/MoodySound'
bucket_name = os.getenv('S3_BUCKET_NAME')


# Start the upload excluding macOS system files to aws s3 bucket
os.system(f'aws s3 cp {data_path} s3://{bucket_name}/data \
    --recursive \
    --exclude "._*" \
    --exclude ".DS_Store"')


