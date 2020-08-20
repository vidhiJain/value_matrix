import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import nav_expert
from plan_network import Model8x8

max_map_size = 8
total_goals = max_map_size ** 2
train_frac = 0.3
val_frac = 0.2
summary_interval = 1

_goals = np.random.randint(max_map_size, size=(total_goals, 2))
# train_goals = _goals[:int(total_goals * train_frac)]
# val_goals = _goals[int(total_goals * train_frac):int(total_goals * (train_frac + val_frac))]
# test_goals = _goals[int(total_goals * (train_frac + val_frac)):]
# np.save('train_goals', train_goals)
# np.save('val_goals', val_goals)
# np.save('test_goals', test_goals)
num_train = int(total_goals * train_frac)
num_val = int(total_goals * val_frac)
num_test = total_goals - (num_train + num_val)

train_goals = _goals[:num_train]
val_goals = _goals[num_train: num_train+num_val]
test_goals = _goals[:-num_test]
np.save('train_goals', train_goals)
np.save('val_goals', val_goals)
np.save('test_goals', test_goals)

model = Model8x8()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #, momentum=0.1)
# scheduler = 
# batch_size
bs = 4
avg_train_loss = []
avg_val_loss = []
for epoch in range(30):
    epoch_loss = []
    # Train
    model.train()
    
    for i in range(train_goals.shape[0]//bs):
        optimizer.zero_grad()
        batch = train_goals[i*bs: (i+1)*bs]
        env = torch.zeros((bs, 1, max_map_size, max_map_size))
        env[np.arange(bs), 0, batch[:, 0], batch[:, 1]] = 2.

        path_matrices = torch.zeros((bs, 1, max_map_size, max_map_size))
        for j in range(bs):
            path_matrices[j] = torch.tensor(nav_expert.get_path_matrix(
                    env[j, 0], batch[j, 0], batch[j, 1], 
                    reward=2, discount_factor=0.99, time_penalty=0.01)
                ).float()
        # path_matrix_tensor = torch.tensor(path_matrix.reshape(1, 1, max_map_size, max_map_size)).float()
        out = model(env)

        loss = criterion(out, path_matrices)    
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    print('epoch: ', epoch, 'train loss: ', sum(epoch_loss) / num_train)
    
    # Validate
    model.eval()
    with torch.no_grad():
        val_loss = []

        # for i, j in val_goals:
        # bs = val_goals.shape[0]
        env = torch.zeros((val_goals.shape[0], 1, max_map_size, max_map_size))
        env[np.arange(val_goals.shape[0]), 0, val_goals[:, 0], val_goals[:, 1]] = 2
        # breakpoint()
        path_matrices = torch.zeros((val_goals.shape[0], 1, max_map_size, max_map_size))
        for j in range(0, val_goals.shape[0]):
            path_matrices[j] = torch.tensor(nav_expert.get_path_matrix(
                env[j, 0], val_goals[j, 0], val_goals[j, 1], 
                reward=2, discount_factor=0.99, time_penalty=0.01)
            ).float()
        # path_matrix = nav_expert.get_path_matrix(env[, 0], i, j, reward=2, discount_factor=0.99, time_penalty=0.01)
        # path_matrix_tensor = torch.tensor(path_matrix.reshape(1, 1, max_map_size, max_map_size)).float()
        out = model(env)
        loss = criterion(out, path_matrices)   
        val_loss.append(loss.item())

    print('epoch: ', epoch, 'val loss: ', sum(val_loss) / num_val)       
    
    # plot_check(out, path_matrix)
    if epoch % summary_interval == 0:
        torch.save(model.state_dict(), 'model8x8.pt')
        avg_train_loss.append(sum(epoch_loss) / num_train)
        avg_val_loss.append(sum(val_loss) / num_val)

# # Test 

# model.eval()
# with torch.no_grad():
#     test_loss = []

#     for i, j in test_goals:
#         env = torch.zeros((1, 1, max_map_size, max_map_size))
#         env[0, 0, i, j] = 2
#         path_matrix = nav_expert.get_path_matrix(env[0, 0], i, j, reward=2, discount_factor=0.99, time_penalty=0.01)
#         path_matrix_tensor = torch.tensor(path_matrix.reshape(1, 1, max_map_size, max_map_size)).float()
#         out = model(env)
#         loss = criterion(out, path_matrix_tensor)   
#         test_loss.append(loss.item())


breakpoint()
plt.plot(avg_train_loss, label='train')
plt.plot(avg_val_loss, label='val')
plt.legend()
plt.ylabel('MSE loss')
plt.xlabel('Epoch number')
plt.title('Train and validation loss')
plt.show()

# TODO #: put clutter to it marlgrid
# TODO : random submap masks



