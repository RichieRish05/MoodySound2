import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MoodyConvNet
from dataset import MoodyDataset
from sklearn.model_selection import KFold

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()



def train_one_epoch(epoch, model, trainloader, optimizer, loss_function, fold, device):
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
            torch.cuda.empty_cache()  
        
            # Free up memory
            del outputs, spectrograms, targets, loss
    
    # Save the model after each epoch
    print(f'Saving model after epoch {epoch+1}')
    save_model(model, epoch, fold)


def save_model(model, epoch, fold):
    save_path = f"/content/Drive/MyDrive/MoodyModelsTest/model_fold_{fold}_epoch_{epoch+1}.pth"
    try:
        torch.save(model.state_dict(), save_path)
    except Exception as e:
        print(f"Error saving model: {e}")




def evaluate_model(model, testloader, loss_function, fold, device):
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

            # Free up memory
            del outputs, spectrograms, targets, mse
            if index % 500 == 0:
                torch.cuda.empty_cache() 
        
        avg_mse = total_mse / total_samples


        return avg_mse
        
        




if __name__ == "__main__":

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Hyper parameters (USE RAYTUNE)
    spectrogram_size = ([1, 128, 1292])
    num_epochs = 32
    batch_size = 100
    learning_rate = 3e-4
    weight_decay = 1e-5 
    k_folds = 5
    loss_function = nn.MSELoss()


    # MoodySound Dataset
    config = "content/MoodySound/data/metadata.csv"
    dataset = MoodyDataset(config=config)


    # Define the k-fold cross validation split
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Initialize the results dictionary
    results = {}

    # Iterate over the k-folds
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        
        # Randomly sample the subsets of the data provided by the k folds
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Load in the training data
        trainloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=train_subsampler
        )

        # Load in the testing data 
        testloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=test_subsampler
        )

        # Initialize the model
        model = MoodyConvNet().to(device)

        # Reset the weights
        model.apply(reset_weights)

        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr = learning_rate)

        # Train the model
        for epoch in range(0, num_epochs):
            print(f'Training epoch {epoch+1}')
            train_one_epoch(epoch=epoch,
                            fold=fold,
                            model=model, 
                            trainloader=trainloader, 
                            optimizer=optimizer, 
                            loss_function=loss_function,
                            device=device)
            

        # Training process is complete for this fold
        print(f'Training process is complete for fold {fold}')
        
        # Print about testing
        print('Starting testing')


        # Evaluate the model
        avg_mse = evaluate_model(model=model, 
                                testloader=testloader, 
                                loss_function=loss_function, 
                                fold=fold,
                                device=device)
        
        results[fold] = avg_mse
        
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value}')
            sum += value
        print(f'Average: {sum/len(results.items())}')