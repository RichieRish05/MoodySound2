import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision import models
from itertools import chain

# IMPLEMENT RESNET18 TRANSFER LEARNING

class Resnet18(nn.Module):


  def __init__(self, dropout_rate=0.15):
    super().__init__()
    self.dropout_rate = dropout_rate

    # Load Resnet18 model
    self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    self.replace_first_conv_layer()
    self.freeze_resnet()

    # Modified classifier for 8 emotion classes
    self.resnet.fc = nn.Sequential(
        nn.Dropout(self.dropout_rate),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(self.dropout_rate),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256), 
        nn.ReLU(),
        nn.Dropout(self.dropout_rate),
        nn.Linear(256, 8),
        nn.LogSoftmax(dim=1)
    )


  def replace_first_conv_layer(self):
    """Convert first conv layer to handle single-channel spectrograms"""
    self.resnet.conv1 = nn.Conv2d(
        in_channels=1,  # Spectrogram is single channel
        out_channels=64, 
        kernel_size=7, 
        stride=2, 
        padding=3, 
        bias=False
    )

    # Initialize the weights by averaging the pretrained RGB weights
    with torch.no_grad():
        original_weights= self.resnet.conv1.weight.data
        averaged_weights = torch.mean(original_weights, dim=1, keepdim=True)
        self.resnet.conv1.weight.data = averaged_weights
  

  def freeze_resnet(self):
    # Only freeze early layers (feature extractors for basic shapes/patterns)
    # and allow later layers to adapt to the new domain
    for param in self.resnet.parameters():
      param.requires_grad = False
    
    # Unfreeze from layer2 onwards for more extensive fine-tuning
    # since spectrograms are very different from natural images
    for layer in [self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
      for param in layer.parameters():
        param.requires_grad = True

  def forward(self, x):
    return self.resnet(x)

  
  def get_tunable_layers_parameters(self):
    return chain(
        self.resnet.layer2.parameters(),
        self.resnet.layer3.parameters(),
        self.resnet.layer4.parameters()
    )

  def get_classifier_parameters(self):
    return self.resnet.fc.parameters()
  
if __name__ == '__main__':
    import numpy as np
    from torch import optim




    model = Resnet18()
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, 1, 128, 1292)
        output = model(x)
        print(output)

    # optimizer = optim.Adam(model.get_tunable_layers_parameters(), lr=0.001)

    # print(model.parameters())
    # print(model.get_tunable_layers_parameters())
    # print(model.get_classifier_parameters())

