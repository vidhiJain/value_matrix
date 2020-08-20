import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import nav_expert
from plan_network import Model51x51
from utils import plot_check
from data import PathPlanData

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
map_max_size = 25
test_goals = np.load('grid_goals_25.npz')['val'][:3]
data = PathPlanData(test_goals, map_max_size, reward=2)
test_dataloader = DataLoader(data, batch_size=1, shuffle=False)

model = Model51x51().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #, momentum=0.1)

PATH = 'model25_7533.pt'
# model.load_state_dict(torch.load(PATH))
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
criterion = nn.MSELoss()


for env, path_matrix in test_dataloader:
        env = env.to(device)
        path_matrix = path_matrix.to(device)        
        
        out = model(env)

        loss = criterion(out, path_matrix)  
        print(loss.item())

        plot_check(out.detach().cpu().numpy()[0, 0], path_matrix.cpu().numpy()[0, 0])

# for i, j in test_goals:
#     env = torch.zeros((1, 1, 8, 8))
#     env[0, 0, i, j] = 2
#     path_matrix = nav_expert.get_path_matrix(env[0, 0], i, j, reward=2, discount_factor=0.99, time_penalty=0.01)
#     path_matrix_tensor = torch.tensor(path_matrix.reshape(1, 1, 8, 8)).float()
#     out = model(env)
    
#     plot_check(out, path_matrix)

