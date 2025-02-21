from hyperparameter_search import train_one_epoch, evaluate_model
import torch
from torch.utils.data import DataLoader
from dataset import MoodyDataset
from custom_model import MoodyConvNet
from torch import nn
import tempfile
import boto3
import os
from dotenv import load_dotenv
# TODO: ADD CHECKPOINTING AND GET CONFIG FROM S3 AND CHECKPOINTING

load_dotenv()

def load_data(csv_path, batch_size):
    dataset = MoodyDataset(config=csv_path)

    torch.manual_seed(42)
    
    dataset_size = len(dataset)


    train_data_size =int(0.9*dataset_size)
    test_data_size = dataset_size - train_data_size


    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_data_size, test_data_size],
        generator = torch.Generator().manual_seed(42)
    )

    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )  

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )


    return trainloader, testloader

def get_config(path_to_best_model):
    s3 = boto3.client('s3')
    s3.download_file(
        Bucket=os.getenv('BUCKET_NAME'),
        Key=path_to_best_model,
        Filename='best_model.pth'
    )
    with open(path_to_best_model, 'r') as f:
        config = torch.load('best_model.pth', map_location=torch.device('cpu'))['config']
    return config

def upload_file_to_s3(file_path, key):
    s3 = boto3.client('s3')

    bucket_name = os.getenv('BUCKET_NAME')
    s3.upload_file(file_path, bucket_name, key)

def checkpoint_model(model, epoch, avg_mse, config):
    with tempfile.NamedTemporaryFile() as temp_file:
        torch.save(
            {'state_dict': model.state_dict(), 
            'epoch': epoch,
            'loss': avg_mse,
            'config': config,
            }, temp_file.name)
        
        upload_file_to_s3(temp_file.name, 'final_model/best_model.pth')

def train_final_model():
    config = get_config()
    trainloader, testloader = load_data(config['csv_path'], config['batch_size'])

    model = MoodyConvNet()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    loss_function = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_mse = float('inf')
    for epoch in range(config['epochs']):
        train_one_epoch(model, trainloader, optimizer, loss_function, device)
        avg_mse = evaluate_model(model, testloader, loss_function, device)


        if avg_mse < best_mse:
            best_mse = avg_mse
            checkpoint_model(model, epoch, avg_mse, config)
