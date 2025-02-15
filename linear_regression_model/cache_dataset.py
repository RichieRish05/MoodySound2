import boto3
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
load_dotenv()



class S3Downloader:

    def __init__(self, bucket, cache_dir):
        self.bucket = bucket
        self.cache_dir = cache_dir
        self.s3_client = boto3.client('s3')



    def create_cache_dir_locally(self):
        os.makedirs(self.cache_dir, exist_ok=True)

    
    def download_a_single_file(self, file):
        destination = os.path.join(self.cache_dir, file['Key'])
        
        try:
            self.s3_client.download_file(
                Bucket = self.bucket,
                Key = file['Key'],
                Filename = destination 
            )
            print(f"Downloaded {file['Key']} to {destination}")
        except Exception as e:
            print(f"Error downloading {file['Key']}: {e}")
    
    def parallel_download(self, page):
        with ThreadPoolExecutor(max_workers=20) as executor:
            executor.map(self.download_a_single_file, page['Contents'])


    def download_dir(self, dist = 'data/'):
        self.create_cache_dir_locally()

        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket, Delimiter = '/', Prefix = dist)


        for page in page_iterator:
            if 'CommonPrefixes' in page:
                for prefix in page['CommonPrefixes']:
                    self.download_dir(prefix['Prefix'])
                    
            if 'Contents' in page and page['Contents']:
                first_key = page['Contents'][0]['Key']
                destination_dir = os.path.join(self.cache_dir, os.path.dirname(first_key))
                os.makedirs(destination_dir, exist_ok=True)

                self.parallel_download(page)
                        




def main():
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    CACHE_DIR = '/workspace'
    s3_downloader = S3Downloader(bucket=BUCKET_NAME, cache_dir=CACHE_DIR)
    s3_downloader.download_dir()


if __name__ == '__main__':
    main()










