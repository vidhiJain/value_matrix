import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import nav_expert
from plan_network import Model
from utils import plot_check

_goals = np.random.randint(4, size=(40, 2))
# train_goals = _goals[:20]
test_goals = _goals[20:]

model = Model()
# PATH = 'model.pt'
PATH = 'tmp1_model.pt'
model.load_state_dict(torch.load(PATH))

for i, j in test_goals:
    env = torch.zeros((1, 1, 4, 4))
    env[0, 0, i, j] = 2
    path_matrix = nav_expert.get_path_matrix(env[0, 0], i, j, reward=2, discount_factor=0.99, time_penalty=0.01)
    path_matrix_tensor = torch.tensor(path_matrix.reshape(1, 1, 4, 4)).float()
    out = model(env)
    
    plot_check(out, path_matrix)
