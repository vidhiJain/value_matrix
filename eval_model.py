import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import nav_expert
from plan_network import Model
from utils import plot_check

test_goals = np.load('test_goals.npy')

model = Model()
PATH = 'model.pt'
model.load_state_dict(torch.load(PATH))

for i, j in test_goals:
    env = torch.zeros((1, 1, 8, 8))
    env[0, 0, i, j] = 2
    path_matrix = nav_expert.get_path_matrix(env[0, 0], i, j, reward=2, discount_factor=0.99, time_penalty=0.01)
    path_matrix_tensor = torch.tensor(path_matrix.reshape(1, 1, 8, 8)).float()
    out = model(env)
    
    plot_check(out, path_matrix)
