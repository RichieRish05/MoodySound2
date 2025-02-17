import boto3
import torch
import os
import tempfile
import ray.cloudpickle as pickle


def save_pkl_as_pth(pkl_path, bucket_name, key):
    with open(pkl_path, "rb") as f:
        checkpoint_data = pickle.load(f)
        
    # Create temporary file to save the checkpoint as a pth file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        # Save as .pth file
        torch.save({
            'epoch': checkpoint_data['epoch'],
            'model_state_dict': checkpoint_data['model_state_dict'],
            'optimizer_state_dict': checkpoint_data['optimizer_state_dict'],
            'loss': checkpoint_data['loss'],
            'config': checkpoint_data['config']
        }, temp_file.name)
            
        # Close the file before uploading
        temp_file.close()
            
        # Upload to S3
        s3 = boto3.client('s3')
        s3.upload_file(
            Bucket=bucket_name,
            Key=key,
            Filename=temp_file.name
        )
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        print(f"Successfully uploaded checkpoint to s3://{bucket_name}/{key}")


def save_best_checkpoint_in_s3_as_pth(bucket_name):
    s3 = boto3.client('s3')
    key = get_best_checkpoint_path_in_s3(bucket_name, 'ray_results/MoodyConvNet/best_checkpoint.txt')

    try:
        s3.download_file(
            Bucket=bucket_name,
            Key=key + "/data.pkl",
            Filename='best_checkpoint.pkl'
        )

        save_pkl_as_pth('best_checkpoint.pkl', bucket_name, 'ray_results/best_model.pth')

        os.remove('best_checkpoint.pkl')
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise


def get_best_checkpoint_path_in_s3(bucket_name, key):
    s3 = boto3.client('s3')

    try:
        with tempfile.NamedTemporaryFile(delete=True) as temp:
            s3.download_file(
                Bucket=bucket_name,
                Key=key,
                Filename=temp.name
            )

            with open(temp.name, 'r') as f:
                return f.read()[len(bucket_name) + 1:]

    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise

if __name__ == "__main__":
    save_best_checkpoint_in_s3_as_pth(os.getenv('S3_BUCKET_NAME'))