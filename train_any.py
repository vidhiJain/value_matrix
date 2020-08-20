import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import PathPlanData
import nav_expert
from plan_network import Model51x51

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

max_map_size = 25
total_goals = max_map_size ** 2
train_frac = 0.4
val_frac = 0.1
summary_interval = 10
bs = 64
PATH = 'model' + str(max_map_size) + '_7533.pt'

_goals = np.random.randint(max_map_size, size=(total_goals, 2))
num_train = int(total_goals * train_frac)
num_val = int(total_goals * val_frac)
num_test = total_goals - (num_train + num_val)

train_goals = _goals[:num_train]
val_goals = _goals[num_train: num_train+num_val]
test_goals = _goals[-num_test:]

np.savez('grid_goals_{}'.format(max_map_size), train=train_goals, val=val_goals, test=test_goals)


train_dataset = PathPlanData(train_goals, max_map_size, reward=2)
val_dataset = PathPlanData(val_goals, max_map_size, reward=2)

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)


model = Model51x51().to(device) # use apt model with sufficient layers when changing the gridsize
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #, momentum=0.1)
# scheduler = 

avg_train_loss = []
avg_val_loss = []

for epoch in range(0, 51):
    epoch_loss = []
    # Train
    model.train()
    
    for env, path_matrix in tqdm(train_dataloader):
        optimizer.zero_grad()

        env = env.to(device)
        path_matrix = path_matrix.to(device)        

        out = model(env)

        loss = criterion(out, path_matrix)    
        loss.backward()
        optimizer.step()

        if epoch % summary_interval == 0:
            epoch_loss.append(loss.item())

    
    if epoch % summary_interval == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)

        print('epoch: ', epoch, 'train loss: ', sum(epoch_loss) / num_train)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = []
            for env, path_matrix in tqdm(val_dataloader):

                env = env.to(device)
                path_matrix = path_matrix.to(device)

                out = model(env)
                loss = criterion(out, path_matrix)   
                val_loss.append(loss.item())

            print('epoch: ', epoch, 'val loss: ', sum(val_loss) / num_val)       
        
            avg_train_loss.append(sum(epoch_loss) / num_train)
            avg_val_loss.append(sum(val_loss) / num_val)
        # torch.save(model.state_dict(), 'model' + str(max_map_size) + '.pt')

breakpoint()
plt.plot(avg_train_loss, label='train')
plt.plot(avg_val_loss, label='val')
plt.legend()
plt.ylabel('MSE loss')
plt.xlabel('Epoch number (x {})'.format(summary_interval))
plt.title('Train and validation loss')
plt.savefig('loss_{0}x{0}'.format(max_map_size))
# plt.show()

# TODO #: put clutter to it marlgrid
# TODO : random submap masks




