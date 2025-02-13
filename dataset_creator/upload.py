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
        self.files_to_upload = self.get_files_to_upload()

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
            print(str(file))
            print(file.name)
            return f'data/{file.name}'


    def get_files_to_upload(self):
        files_to_upload = []
        
        
        for file in self.data_path.rglob('*'):
            if file.is_file() and self.validate_file(file):
                local_path = str(file)
                s3_key = self.create_s3_key(file)
                files_to_upload.append((local_path, s3_key))
            
        return files_to_upload
    
    def upload_file_to_s3(self, upload_info):
        local_path, s3_key = upload_info
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
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(self.upload_file_to_s3, self.files_to_upload)

            successful_uploads = sum(1 for result in results if result)
            total_files = len(self.files_to_upload)
            failed_uploads = total_files - successful_uploads
            print(f'Uploaded {successful_uploads} files')
            print(f'Failed to upload {failed_uploads} files')
            print(f'Total files: {total_files}')


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
    data_path = '/Volumes/Drive/MoodySound/test_data'
    # s3_uploader = S3Uploader(data_path, bucket_name)
    # s3_uploader.upload_directory_to_s3()

    # file_name = 'Mouthtrap_30s_matrix.npy'
    # test_upload(bucket_name, file_name)