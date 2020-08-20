import torch

from plan_network import Model 

model = Model()
PATH = 'model.pt'
model.load_state_dict(torch.load(PATH))

# given a test scenario with random goal location, and random agent position
# agent takes step to its neighbour that maximizes the value
# model receives the training reward signal when the agent reaches the goal or exhausts max_steps 
# reward is computed by +2 for reaching the goal, -1 for obstacles, -0.01 as time penalty for every step
# plot mean reward by curriculum training (curriculum is the increasing distance between the goal and the agent)

