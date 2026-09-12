'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn as nn


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Networks for the NDP
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

# Graph Cellular Automata model employed for the graph convolution
class FixedMLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        input_size = config['graph_n_inputs']
        hidden_size = config['fixed_mlp_hidden_size']
        output_size = config['graph_n_outputs']

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)

    def update_mlp_weights(self, parameter_vector):
        parameter_vector = np.asarray(parameter_vector, dtype=np.float32)
        parameter_tensor = torch.from_numpy(parameter_vector)

        torch.nn.utils.vector_to_parameters(
            parameter_tensor,
            self.parameters(),
        )

    def get_total_number_of_mlp_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_activations(self):
        pass

    def reset_weights(self):
        pass

    def summary(self):
        print('-------------------------------------')
        print('Fixed MLP')
        print('-------------------------------------')
        print(self.model)
        print(f'Number of parameters = {self.get_total_number_of_mlp_parameters()}')
        print('-------------------------------------\n')