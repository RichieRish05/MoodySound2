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



# Device configuration
device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')


# Hyper parameters
spectrogram_size = ([1, 128, 1292])
num_epochs = 32
batch_size = 100
learning_rate = 3e-4
weight_decay = 1e-3 # Change to 1e-5
k_folds = 5
loss_function = nn.MSELoss()


# MoodySound Dataset

config = "/Volumes/Drive/MoodySound/data/metadata.csv"
dataset = MoodyDataset(config=config)


# Define the k-fold cross validation split

kfold = KFold(n_splits=k_folds, shuffle=True)

results = {}

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
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = learning_rate, 
                                  weight_decay=weight_decay)

    for epoch in range(0, num_epochs):
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

            # Zero the gradient
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
                print('Loss after mini-batch %5d: %.3f' %
                    (index + 1, current_loss / 500))
                current_loss = 0.0
        
    # Process is complete.

    print('Training process has finished. Saving trained model.')
    
    # Save the model
    save_path = f"/Volumes/Drive/MoodySound/models/model_fold_{fold}.pth"
    torch.save(model.state_dict(), save_path)

    # Print about testing
    print('Starting testing')

    # Evaluation for this fold
    total_mse= 0.0
    total_samples = 0

    with torch.no_grad():
        model.eval()

        for index, data in enumerate(testloader, 0):
            # Collect the spectrograms and associated mood vectors
            spectrograms, targets = data
            spectrograms, targets = spectrograms.to(device), targets.to(device) 

            # Generate outputs
            outputs = model(spectrograms)
 
            loss = loss_function(outputs, targets)

            # Set total loss
            mse = loss_function(outputs, targets).item()
            total_mse+= mse * targets.size(0)
            total_samples += targets.size(0)
        
        fold_mse = total_mse / total_samples
        print(f'MSE for fold {fold}: {fold_mse:.4f}')
        results[fold] = fold_mse
    
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')