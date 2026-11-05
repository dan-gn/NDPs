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

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        else:
            hidden_size = list(hidden_size)

        self.layer_sizes = [input_size, *hidden_size, output_size]
        n_layers = len(self.layer_sizes) # Including input, hidden and output

        layers = []
        for index in range(n_layers - 1):
            layers.append(nn.Linear(self.layer_sizes[index], self.layer_sizes[index+1]))

            if index < n_layers - 2:
                layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)


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

        self.n_inputs = self.layer_sizes[0]
        self.hidden_sizes = self.layer_sizes[1:-1]
        self.n_hidden = sum(self.hidden_sizes)
        self.n_outputs = self.layer_sizes[-1]


        # Evolved eta, A, B, C, D for every neuron.
        self.hebbian_coefficients = nn.Parameter(torch.zeros(sum(self.layer_sizes), 5))

        # This get the indices of the linear layers skipping the activation function
        self.linear_indices = [index for index, layer in enumerate(self.model) if isinstance(layer, nn.Linear)]

        for layer_number, model_index in enumerate(self.linear_indices):
            self.register_buffer(f'current_weights_{layer_number}', self.model[model_index].weight.detach().clone())


    def update_mlp_weights(self, parameter_vector):
        super().update_mlp_weights(parameter_vector)
        self.reset_weights()

    @torch.no_grad()
    def reset_weights(self):
        for layer_number, model_index in enumerate(self.linear_indices):
            current_weights = getattr(self, f'current_weights_{layer_number}')
            current_weights.copy_(self.model[model_index].weight)

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
    def apply_hebbian_update(self, activations):
        coefficients = torch.split(self.hebbian_coefficients, self.layer_sizes, dim=0)

        deltas = [self.compute_hebbian_delta(activations[index], activations[index+1], coefficients[index], coefficients[index+1]) for index in range(len(self.linear_indices))]

        for layer_number, delta in enumerate(deltas):
            current_weights = getattr(self, f'current_weights_{layer_number}')
            current_weights.add_(delta.T).clamp_(-1.0, 1.0)


    @torch.no_grad()
    def forward(self, x):
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError('HebbianFixedMLP expects one observation at a time.')

        activations = [x]

        pre = x
        for layer_number, model_index in enumerate(self.linear_indices):
            layer = self.model[model_index]
            current_weights = getattr(self, f'current_weights_{layer_number}')

            post = F.linear(pre, current_weights, layer.bias)
            pre = torch.tanh(post)

            activations.append(pre)


        self.apply_hebbian_update(activations)

        return post

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
