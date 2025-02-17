import numpy as np
import boto3
from dotenv import load_dotenv
import os
import torch
from model import MoodyConvNet

s = np.load("/Users/rishi/MoodySound2/test/data/spectrograms/Bang_Bang_(My_Baby_Shot_Me_Down)_10s_matrix.npy")
t = np.load("/Users/rishi/MoodySound2/test/data/targets/Bang_Bang_(My_Baby_Shot_Me_Down)_10s_target.npy")
s = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)

print(t)



load_dotenv()


model = MoodyConvNet()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu')['model_state_dict'])

with torch.no_grad():
    model.eval()

    output = model(s)
    print(output)

