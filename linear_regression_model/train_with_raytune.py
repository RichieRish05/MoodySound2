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
        shuffle=True
    ) 

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True
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


        return avg_mse
        
        


def train_model(config):
    """
    This is the training function that will be used by raytune. 
    """

    # Get hyperparameters from the config
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    dropout_rate = config['dropout_rate']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    trainloader, testloader, _ = load_data(config, batch_size)


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
                                testloader=testloader, 
                                loss_function=loss_function, 
                                device=device)
        
        
        tune.report(val_loss=avg_mse)


def main():
    config = {
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        'batch_size': [32,64,128,256],
        'num_epochs': 32,
        'dropout_rate': tune.uniform(0.1, 0.5),
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

    ray.init()

    tuner = tune.Tuner(
        train_model,
        param_space=config,
        

        tune_config = tune.TuneConfig(
            metric = "val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=20,
            search_alg=search_alg
        ),



        run_config = tune.RunConfig(
            storage_path = "/content/drive/MyDrive/MoodyModelsTest",
            name = "MoodyConvNet",
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,  # Only keep the best checkpoint
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min"
            )
        )
    )


    results = tuner.fit()
    
    # Print results
    best_trial = results.get_best_trial("val_loss", "min")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    
    # Save best model
    best_checkpoint = best_trial.checkpoint.value
    best_model_metadata = torch.load(os.path.join(best_checkpoint, "checkpoint"))
    best_model = MoodyConvNet()
    best_model.load_state_dict(best_model_metadata["model_state_dict"])
    torch.save(best_model.state_dict(), 
              "/content/drive/MyDrive/MoodyModelsTest/best_model.pth")



























if __name__ == "__main__":
    main()