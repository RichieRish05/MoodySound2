import torch
import ray
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MoodyConvNet
from dataset import MoodyDataset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
from ray.tune.search.optuna import OptunaSearch  
import boto3
from dotenv import load_dotenv
import tempfile
import ray.cloudpickle as pickle
from ray.train import Checkpoint

# Load the environment variables
load_dotenv()
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')



def load_data(config, batch_size):
    dataset = MoodyDataset(config=config)

    torch.manual_seed(42)
    
    dataset_size = len(dataset)


    train_data_size =int(0.8*dataset_size)
    test_data_size = int(0.1*dataset_size)
    val_data_size = dataset_size - train_data_size - test_data_size


    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_data_size, test_data_size, val_data_size],
        generator = torch.Generator().manual_seed(42)
    )

    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )  

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )


    return trainloader, testloader, val_loader





def train_one_epoch(epoch, model, trainloader, optimizer, loss_function, device):
    model.train()

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the train data loader
    for index, data in enumerate(trainloader, 0):
            
        # Collect the spectrograms and associated mood vectors
        spectrograms, targets = data
        spectrograms, targets = spectrograms.to(device), targets.to(device) 

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(spectrograms)

        # Compute the loss
        loss = loss_function(outputs, targets)

        # Backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
        if index % 500 == 499:
            print(f'Epoch {epoch+1} - Loss after mini-batch {index+1}: {current_loss/500}')
            current_loss = 0.0

            # Free up memory
            torch.cuda.empty_cache()  
            del outputs, spectrograms, targets, loss
    






def evaluate_model(model, testloader, loss_function, device):
    print("\nEvaluating model...")
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        model.eval()

        for index, data in enumerate(testloader, 0):
            spectrograms, targets = data
            spectrograms, targets = spectrograms.to(device), targets.to(device) 

            outputs = model(spectrograms)
            mse = loss_function(outputs, targets).item()
            total_mse += mse * targets.size(0)
            total_samples += targets.size(0)


            if index % 500 == 0:
                del outputs, spectrograms, targets, mse
                torch.cuda.empty_cache() 
        
        avg_mse = total_mse / total_samples

        print(f"Evaluation complete - Average MSE: {avg_mse}")


        return avg_mse
        
        


def train_model(config):
    """
    This is the training function that will be used by raytune. 
    """

    # Get the trial id
    trial = ray.train.get_context()
    trial_id = trial.get_trial_id()
    
    print(f"Trial ID: {trial_id}")


    print(f"\n=== Starting hyperparameter tuning for {trial_id} ===")

    print(f"Training model with config: {config}")

    # Get hyperparameters from the config
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    dropout_rate = config['dropout_rate']
    csv_path = config['csv_path']  


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")


    print("Loading data...")
    # Pass csv_path directly instead of nested in another dict
    trainloader, _ , val_loader = load_data(config=csv_path, batch_size=batch_size)
    print(f"Data loaded. Train batches: {len(trainloader)}, Val batches: {len(val_loader)}")


    # Initialize the model
    model = MoodyConvNet(dropout_rate=dropout_rate).to(device)


    optimizer = torch.optim.Adam(model.parameters(), 
                                lr = learning_rate, 
                                weight_decay = weight_decay)
    
    loss_function =  nn.MSELoss()

    for epoch in range(num_epochs):
        train_one_epoch(epoch=epoch, 
                        model=model, 
                        trainloader=trainloader, 
                        optimizer=optimizer, 
                        loss_function=loss_function, 
                        device=device)
        
        # Evaluate the model
        avg_mse = evaluate_model(model=model, 
                                testloader=val_loader, 
                                loss_function=loss_function, 
                                device=device)
        
        # Create checkpoint
        checkpoint_info = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_mse,
            'config': config,
            'trial_id': trial_id  # Include trial_id in checkpoint info
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:
                pickle.dump(checkpoint_info, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            
            # Report metrics to Ray Tune
            tune.report(
                metrics={"val_loss": avg_mse},
                checkpoint=checkpoint
            )


def main():
    # Define the configuration space for the hyperparameters
    config = {
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        'batch_size': tune.choice([128, 256, 512]),
        'num_epochs': 2,
        'dropout_rate': tune.uniform(0.1, 0.5),
        'csv_path': '/Users/rishi/MoodySound2/test/data/shuffled_metadata.csv'  
    }

    scheduler = ASHAScheduler(
        max_t=config['num_epochs'], # Total number of epochs to run
        grace_period=2, # Number of epochs to wait before cutting any models out
        reduction_factor=2 # Cut half of the models
    )

    search_alg = OptunaSearch(
        metric = "val_loss",
        mode = "min",
    )

    print("\nInitializing Ray...")
    ray.init()
    print("Starting tuning...")

    tuner = tune.Tuner(
        train_model,
        param_space=config,
        
        # Tune config TESTING
        tune_config = tune.TuneConfig(
            metric = "val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=2,
            search_alg=search_alg,
        ),


        # Run config TESTING
        run_config = tune.RunConfig(
            storage_path = f"s3://{BUCKET_NAME}/ray_results/",
            #storage_path = "/Users/rishi/MoodySound2/test/ray_results",
            name = "MoodyConvNet",
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,  # Only keep the best checkpoint
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min"
            )
        )
    )


    results = tuner.fit()
    save_best_checkpoint_path_in_s3(results)
   #save_best_checkpoint_in_s3_as_pth(BUCKET_NAME, 'best_checkpoint.txt')


def save_best_checkpoint_path_in_s3(results):
    best_trial = results.get_best_result("val_loss", "min")
    print(f"Best trial config: {best_trial.config}")
    
    # Save the best checkpoint to S3
    best_checkpoint = best_trial.checkpoint.path
    with open('best_checkpoint.txt', 'w') as f:
        f.write(best_checkpoint)


    s3 = boto3.client('s3')
    s3.upload_file(
        Bucket=BUCKET_NAME,
        Key='ray_results/MoodyConvNet/best_checkpoint.txt',
        Filename='best_checkpoint.txt'
    )
    
    os.remove('best_checkpoint.txt')




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


def save_best_checkpoint_in_s3_as_pth(bucket_name, key):
    s3 = boto3.client('s3')

    key = key[len(bucket_name) + 1:]

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


    


















if __name__ == "__main__":
    main()