'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import torch
import torch.nn as nn


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Networks for the NDP
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

# Graph Cellular Automata model employed for the graph convolution
class GraphCellularAutomata(nn.Module):

    def __init__(self, state_dim, hidden_layer_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, state_dim),
            nn.Tanh()
        )

    # def forward(self, node_state, neighbors_states):
    #     if neighbors_states.size()[0] > 1:
    #         neighbors_states_mean = torch.mean(neighbors_states, axis = 0).unsqueeze(axis = 0)
    #         x = torch.cat([node_state, neighbors_states_mean], dim = 1)
    #     else:
    #         x = torch.cat([node_state, neighbors_states], dim = 1)

    #     delta = self.model(x)
    #     new_state = node_state + delta

    #     return new_state
    def forward(self, node_state):
        return self.model(node_state)

# Replication model employed to grow the graph 
class ReplicationModel(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, node_state):
        return self.model(node_state)

# Model to predict weights
class WeightPredictionModel(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, input_node_state, output_node_state):
        x = torch.cat([input_node_state, output_node_state], dim=0)
        return self.model(x)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Networks for the Hebbian NDP (my variant)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

# Model to create edges
class CreateEdgeModel(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, input_node_state, output_node_state):
        x = torch.cat([input_node_state, output_node_state], dim=1)
        return self.model(x)

# Model to remove edges
class RemoveEdgeModel(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, input_node_state, output_node_state):
        x = torch.cat([input_node_state, output_node_state], dim=1)
        return self.model(x)

