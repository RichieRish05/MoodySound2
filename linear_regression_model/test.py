import numpy as np
import boto3
from dotenv import load_dotenv
import os
import torch
from model import MoodyConvNet

s = np.load("/Users/rishi/MoodySound2/test/data/spectrograms/About_a_Girl_10s_matrix.npy")
t = np.load("/Users/rishi/MoodySound2/test/data/targets/About_a_Girl_10s_target.npy")
s = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)

print(t)



load_dotenv()


model = MoodyConvNet()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu')['model_state_dict'])

with torch.no_grad():
    model.eval()

    output = np.array(model(s))

    norm = np.linalg.norm(output, axis=1)
    output = output / norm

    print(output)

# tensor([[0.2687, 0.4549, 0.0752, 0.2391, 0.2974, 0.5133, 0.2151, 0.3664]])