import torch
import ray
import torch.nn as nn
from torch.utils.data import DataLoader
from resnet import Resnet18
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
from custom_model import MoodyConvNet




def load_data(csv_path, batch_size):
    dataset = MoodyDataset(config=csv_path)

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

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )


    return trainloader, testloader, val_loader





def train_one_epoch(epoch, model, trainloader, optimizer, loss_function, device):
    model.train()
    print(f'Starting epoch {epoch+1}')
    current_loss = 0.0

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

        # Print statistics every 500 batches
        current_loss += loss.item()
        if index % 500 == 499:
            print(f'Epoch {epoch+1} - Loss after mini-batch {index+1}: {current_loss/500}')
            current_loss = 0.0

            # Free up memory
            torch.cuda.empty_cache()  
            del outputs, spectrograms, targets
    






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
    # Clear gpu memory
    torch.cuda.empty_cache()

    # Get the trial id
    trial = ray.train.get_context()
    trial_id = trial.get_trial_id()
    


    print(f"\n=== Starting hyperparameter tuning for {trial_id} ===")

    print(f"Training model with config: {config}")

    # Get hyperparameters from the config
    classifier_learning_rate = config['classifier_lr']
    classifier_weight_decay = config['classifier_weight_decay']
    backbone_lr_ratio = config['backbone_lr_ratio']
    tunable_layers_learning_rate = classifier_learning_rate * backbone_lr_ratio
    tunable_layers_weight_decay = classifier_weight_decay * backbone_lr_ratio
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    dropout_rate = config['dropout_rate']
    csv_path = config['csv_path']  


    # Initialize the device as cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")


    # Load the data in separate train and val loaders
    print("Loading data...")
    trainloader, _ , val_loader = load_data(csv_path=csv_path, batch_size=batch_size)
    print(f"Data loaded. Train batches: {len(trainloader)}, Val batches: {len(val_loader)}")


    # Initialize the model
    model = MoodyConvNet(dropout_rate=dropout_rate).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=classifier_learning_rate, weight_decay=classifier_weight_decay)
    
    
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
            'trial_id': trial_id  
        }

        # Create checkpoint object in a temporary directory and report metrics to Ray Tune
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
        'classifier_lr': tune.loguniform(1e-5, 1e-2),
        'classifier_weight_decay': tune.loguniform(1e-6, 1e-4),
        'backbone_lr_ratio': tune.loguniform(0.01, 0.3), 
        'batch_size': tune.choice([32, 64]),
        'num_epochs': tune.choice([16, 24, 32]),
        'dropout_rate': tune.uniform(0.1, 0.5),
        'csv_path': '/workspace/data/metadata.csv'
    }

    scheduler = ASHAScheduler(
        max_t=32,
        grace_period=4,
        reduction_factor=3
    )

    search_alg = OptunaSearch(
        metric = "val_loss",
        mode = "min",
    )

    print("\nInitializing Ray...")
    ray.init(num_gpus=4)
    print("Starting tuning...")

    trainable_with_gpu = tune.with_resources(
        train_model, 
        {"gpu": 1} 
    )

    tuner = tune.Tuner(
        trainable_with_gpu,
        param_space=config,
        
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=8, 
            search_alg=search_alg,
            max_concurrent_trials=4,
        ),


        # Run config
        run_config = tune.RunConfig(
            storage_path = f"s3://{BUCKET_NAME}/ray_results/",
            name = EXPERIMENT_NAME,
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,  
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min"
            )
        )
    )


    results = tuner.fit()
    save_best_checkpoint_path_in_s3(results)


def save_best_checkpoint_path_in_s3(results):
    best_trial = results.get_best_result("val_loss", "min")
    print(f"Best trial config: {best_trial.config}")
    

    best_checkpoint = best_trial.checkpoint.path
    with open('best_checkpoint.txt', 'w') as f:
        f.write(best_checkpoint)


    s3 = boto3.client('s3')

    try:
        # Save the best checkpoint to S3
        s3.upload_file(
            Bucket=BUCKET_NAME,
            Key=f'ray_results/{EXPERIMENT_NAME}/best_checkpoint.txt',
            Filename='best_checkpoint.txt'
        )
        print(f"Best checkpoint uploaded to S3: {BUCKET_NAME}/ray_results/{EXPERIMENT_NAME}/best_checkpoint.txt")
    except Exception as e:
        print(f"Error uploading best checkpoint to S3: {e}")
    finally:
        os.remove('best_checkpoint.txt')



if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    EXPERIMENT_NAME = 'UserDataMoodyConvNet'


    # Run the main function
    main()
    
















