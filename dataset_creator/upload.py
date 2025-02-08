import os



data_path = '/Volumes/Drive/MoodySound/data/'
bucket_name = 'rishitestbucket01'


# Start the upload excluding macOS system files to aws s3 bucket
os.system(f'aws s3 cp {data_path} s3://{bucket_name}/data \
    --recursive \
    --exclude "._*" \
    --exclude ".DS_Store" \
    --multipart-threshold 500MB \
    --multipart-chunksize 500MB')
