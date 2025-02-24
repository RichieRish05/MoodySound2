import boto3
import torch
import os
import tempfile
import ray.cloudpickle as pickle
from dotenv import load_dotenv


class ModelDownloader():

    def __init__(self, bucket_name, experiment_path):
        self.bucket_name = bucket_name
        self.experiment_path = experiment_path
        self.s3 = boto3.client('s3')

    def get_best_checkpoint_path_in_s3(self):
        try:
            with tempfile.NamedTemporaryFile(delete=True) as temp:
                self.s3.download_file(
                    Bucket=self.bucket_name,
                    Key=self.experiment_path + '/best_checkpoint.txt',
                    Filename=temp.name
                )

                with open(temp.name, 'r') as f:
                    return f.read()[len(self.bucket_name) + 1:]

        except Exception as e:
            print(f"Error downloading from S3: {e}")
            raise

    def save_model_as_pth_in_s3(self):
        try:
            key = self.get_best_checkpoint_path_in_s3()
            self.s3.download_file(
                Bucket=self.bucket_name,
                Key=key + '/data.pkl',
                Filename='best_model.pkl'
            )

            self.save_pkl_as_pth('best_model.pkl')
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            raise
        finally:
            os.unlink('best_model.pkl')
    
    def save_pkl_as_pth(self, pkl_path):
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
                Bucket=self.bucket_name,
                Key=self.experiment_path + '/best_model.pth',
                Filename=temp_file.name
            )
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            print(f"Successfully uploaded checkpoint to s3://{self.bucket_name}/{self.experiment_path + '/best_model.pth'}")
    


if __name__ == "__main__":
    load_dotenv()
    bucket = os.getenv('S3_BUCKET_NAME')
    model_downloader = ModelDownloader(bucket, 'ray_results/NormalizedMoodyConvNet')
    model_downloader.save_model_as_pth_in_s3()