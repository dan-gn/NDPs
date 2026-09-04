
'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import jax
import jax.numpy as jnp

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Network using Jax
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


class MLPJax:

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def get_n_parameters(self) -> int:
        input_to_hidden = (self.input_dim + 1) * self.hidden_dim
        hidden_to_output = (self.hidden_dim + 1) * self.output_dim
        return input_to_hidden + hidden_to_output

    def unpack_parameters(self, flat_params:jax.Array) -> tuple:
        pointer = 0

        n = self.hidden_dim * self.input_dim
        w1 = flat_params[pointer:pointer + n].reshape(self.hidden_dim, self.input_dim)
        pointer += n

        n = self.hidden_dim
        b1 = flat_params[pointer:pointer + n]
        pointer += n

        n = self.output_dim * self.hidden_dim
        w2 = flat_params[pointer:pointer + n].reshape(self.output_dim, self.hidden_dim)
        pointer += n

        n = self.output_dim
        b2 = flat_params[pointer:pointer + n]
        pointer += n

        params = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
        }

        return params, pointer

    def forward(self, x:jax.Array, params:dict):
        x = jnp.dot(x, params['w1'].T) + params['b1']
        x = jnp.tanh(x)

        x = jnp.dot(x, params['w2'].T) + params['b2']
        x = jnp.tanh(x)

        return x


class PairMLPJax(MLPJax):

    def forward(self, source_state:jax.Array, target_state:jax.Array, params:dict):
        x = jnp.concatenate([source_state, target_state], axis=-1)
        return super().forward(x, params)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Networks for the NDP
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

# Graph Cellular Automata model employed for the graph convolution
class GraphCellularAutomata(MLPJax):

    def __init__(self, state_dim, hidden_dim):
        super().__init__(input_dim=state_dim, hidden_dim=hidden_dim, output_dim=state_dim)


# Replication model employed to grow the graph 
class ReplicationModel(MLPJax):

    def __init__(self, state_dim, hidden_dim):
        super().__init__(input_dim=state_dim, hidden_dim=hidden_dim, output_dim=1)


# Model to predict weights
class WeightPredictionModel(PairMLPJax):

    def __init__(self, state_dim, hidden_dim):
        super().__init__(input_dim=state_dim * 2, hidden_dim=hidden_dim, output_dim=1)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Networks for the Hebbian NDP (my variant)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

# Model to create edges
class CreateEdgeModel(PairMLPJax):
 
    def __init__(self, state_dim, hidden_dim):
        super().__init__(input_dim=state_dim * 2, hidden_dim=hidden_dim, output_dim=1)


# Model to remove edges
class RemoveEdgeModel(PairMLPJax):

    def __init__(self, state_dim, hidden_dim):
        super().__init__(input_dim=state_dim * 2, hidden_dim=hidden_dim, output_dim=1)
