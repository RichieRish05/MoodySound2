import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision import models



class MoodyConvNet(nn.Module):
  '''
    Simple Convolutional Neural Network
  '''
  def __init__(self, dropout_rate=0.15):
    super().__init__()
    self.dropout_rate = dropout_rate
  
    # Convolutional layers
    self.layers = nn.Sequential(
      # First convolutional layer
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32), 
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 4)),  # Reduce both frequency and time dimensions

      # Second convolutional layer
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 4)),

      # Third convolutional layer
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 4)),
      
      # Fourth convolutional layer
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 4)),

      # Flatten the output
      nn.Flatten(),

      # Dense layers
      nn.Linear(256 * 8 * 5, 512),  # Based on spectrogram dimensions
      nn.ReLU(),
      nn.Dropout(self.dropout_rate),
      
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(self.dropout_rate),
      
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(self.dropout_rate),

      nn.Linear(128, 8), # 8 emotion classes
      nn.Sigmoid()

    )


  def forward(self, x):
    '''Forward pass with normalized output vectors'''
    return self.layers(x)
  
 