'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Fixed MLP to evolve on direct encoding
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

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

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FixedMLP with Neuron-centric Hebbian Learning
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''



class HebbianFixedMLP(FixedMLP):

    def __init__(self, config):
        super().__init__(config)

        self.n_inputs = self.model[0].in_features
        self.n_hidden = self.model[0].out_features
        self.n_outputs = self.model[2].out_features

        # Evolved eta, A, B, C, D for every neuron.
        self.hebbian_coefficients = nn.Parameter(torch.zeros(self.n_inputs + self.n_hidden + self.n_outputs, 5))

        self.register_buffer('current_input_to_hidden_weights', self.model[0].weight.detach().clone())
        self.register_buffer('current_hidden_to_output_weights', self.model[2].weight.detach().clone())

    def update_mlp_weights(self, parameter_vector):
        super().update_mlp_weights(parameter_vector)
        self.reset_weights()

    def reset_weights(self):
        with torch.no_grad():
            self.current_input_to_hidden_weights.copy_(self.model[0].weight)
            self.current_hidden_to_output_weights.copy_(self.model[2].weight)

    # Compute Hebbian delta
    @staticmethod
    def compute_hebbian_delta(pre:torch.Tensor, post:torch.Tensor, source_coeff:torch.Tensor, target_coeff:torch.Tensor):
        eta = (source_coeff[:, None, 0] + target_coeff[None, :, 0]) / 2

        # delta = ((etai+etaj)/2) * (Ai*pre + Bj*post + Ci*Cj*pre*post + Di*Dj)
        delta = eta * (
            source_coeff[:, None, 1] * pre.reshape(-1, 1)     # Ai * pre
            + target_coeff[None, :, 2] * post.reshape(1, -1)  # Bj * post  
            + source_coeff[:, None, 3] * target_coeff[None, :, 3]
            * pre.reshape(-1, 1) * post.reshape(1, -1)  # Ci * Cj * pre * post
            + source_coeff[:, None, 4] * target_coeff[None, :, 4]   # Di * Dj
        )

        return delta
    
    @torch.no_grad()
    def apply_hebbian_update(self, x, hidden, output):
        input_coeff = self.hebbian_coefficients[:self.n_inputs]
        hidden_coeff = self.hebbian_coefficients[self.n_inputs:self.n_inputs + self.n_hidden]
        output_coeff = self.hebbian_coefficients[self.n_inputs + self.n_hidden:]

        input_delta = self.compute_hebbian_delta(x, hidden, input_coeff, hidden_coeff)
        output_delta = self.compute_hebbian_delta(hidden, torch.tanh(output), hidden_coeff, output_coeff)

        self.current_input_to_hidden_weights.add_(input_delta.T).clamp_(-1, 1)
        self.current_hidden_to_output_weights.add_(output_delta.T).clamp_(-1, 1)

    @torch.no_grad()
    def forward(self, x):
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError('HebbianFixedMLP expects one observation at a time.')

        hidden = torch.tanh(F.linear(x, self.current_input_to_hidden_weights, self.model[0].bias))
        output = F.linear(hidden, self.current_hidden_to_output_weights, self.model[2].bias)

        self.apply_hebbian_update(x, hidden, output)

        return output

    def summary(self):
        network_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        hebbian_parameters = self.hebbian_coefficients.numel()

        print('-------------------------------------')
        print('Fixed MLP with Neuron-centric Hebbian Learning')
        print('-------------------------------------')
        print(self.model)
        print(f'Neurons: inputs={self.n_inputs}, hidden={self.n_hidden}, outputs={self.n_outputs}')
        print('-------------------------------------')
        print('Evolved parameters')
        print(f'Initial weights and biases = {network_parameters}')
        print(f'Hebbian coefficients (eta, A, B, C, D per neuron) = {hebbian_parameters}')
        print(f'Total = {self.get_total_number_of_mlp_parameters()}')
        print('-------------------------------------')
