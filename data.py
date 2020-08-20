import torch
from torch.utils.data import Dataset
import nav_expert

class PathPlanData(Dataset):
    def __init__(self, goals, max_map_size, reward):
        self.goals = goals
        self.max_map_size = max_map_size
        self.reward = reward

    def __len__(self):
        return self.goals.shape[0]

    def __getitem__(self, i):
        idx = self.goals[i]
        env = torch.zeros((self.max_map_size, self.max_map_size))
        env[self.goals[i, 0], self.goals[i, 1]] = self.reward

        path_matrix = torch.tensor(nav_expert.get_path_matrix(
                env, self.goals[i, 0], self.goals[i, 1], 
                reward=2, discount_factor=0.99, time_penalty=0.001)
            ).float()

        return env.unsqueeze(0), path_matrix.unsqueeze(0)
    # def get_dataset(goals):
    #     env = torch.zeros((goals.shape[0], 1, max_map_size, max_map_size))
    #     env[np.arange(goals.shape[0]), 0, goals[:, 0], goals[:, 1]] = 2
    #     # breakpoint() 
    #     path_matrices = torch.zeros((goals.shape[0], 1, max_map_size, max_map_size))
    #     for j in range(0, goals.shape[0]):
    #         path_matrices[j] = torch.tensor(nav_expert.get_path_matrix(
    #             env[j, 0], goals[j, 0], goals[j, 1], 
    #             reward=2, discount_factor=0.99, time_penalty=0.01)
    #         ).float()
    #     return env, path_matrices

if __name__=="__main__":
    import numpy as np
    goals = np.array([[6, 0],[1, 2]])
    data = PathPlanData(goals, max_map_size=8, reward=2)
    env, path_matrix = data.__getitem__(0)
    breakpoint()