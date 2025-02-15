import os
from dotenv import load_dotenv
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class S3Uploader:
    def __init__(self, data_path, bucket_name):
        self.data_path = Path(data_path)
        self.s3client = boto3.client('s3')
        self.bucket_name = bucket_name


    def validate_file(self, file_name):
        path = Path(file_name)
        if path.name.startswith('.'):
            return False
        if path.name.startswith('__'):
            return False
        if 'DS_Store' in path.name:
            return False
        return True

    def create_s3_key(self, file):
        relative_path = file.relative_to(self.data_path)
        if 'spectrograms' in relative_path.parts:
            return f'data/spectrograms/{file.name}'
        if 'targets' in relative_path.parts:
            return f'data/targets/{file.name}'
        else:
            return f'data/{file.name}'


    def upload_file_to_s3(self, file):
        if not file.is_file() or not self.validate_file(file):
            return False

        local_path = str(file)
        s3_key = self.create_s3_key(file)
        
        try:
            self.s3client.upload_file(
                Bucket=self.bucket_name,
                Filename=local_path,
                Key=s3_key
            )

            print(f'Uploaded {local_path} to {s3_key}')
            return True

        except Exception as e:
            print(f'Error uploading {local_path} to {s3_key}: {e}')
            return False
    
    def upload_directory_to_s3(self):
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(self.upload_file_to_s3, self.data_path.rglob('*')))

            successful_uploads = sum(1 for result in results if result)
            print(f'Uploaded {successful_uploads} files')


def test_upload(bucket_name, file_name):
    s3_client = boto3.client('s3')

    if 'target' in file_name:
        file_path = f'data/targets/{file_name}'
    else:
        file_path = f'data/spectrograms/{file_name}'
    
    s3_client.download_file(
        Bucket=bucket_name,
        Key=file_path,
        Filename=f'dataset_creator/test.npy'
    )

    data = np.load(f'dataset_creator/test.npy')
    print(data)
    print(data.shape)
    print(data.dtype)


if __name__ == '__main__':
    load_dotenv()
    bucket_name = os.getenv('S3_BUCKET_NAME')
    data_path = '/Volumes/Drive/MoodySound/data'
    s3_uploader = S3Uploader(data_path, bucket_name)
    print(f"Uploading data to {bucket_name}")
    s3_uploader.upload_directory_to_s3()