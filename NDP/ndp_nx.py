'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch 
import torch.nn as nn
import time
import warnings

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from NDP.ndp_mlps import GraphCellularAutomata, ReplicationModel, WeightPredictionModel
from Graph.graph import Node, Graphnx

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Default parameters
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

DEFAULT_PARAMETERS = {
    'state_dim': 5,
    'weighted_graph_flag': True,
    'initial_graph': 'one_node',
    'add_hidden_node_to_minimal_network': True,    # Required if initial_graph == 'minimal_network'
    'network_extra_thinking': 0,
    'initial_node_state_mode': 'coevolve', 
    'shared_initial_node_state': None,
    'noise_while_growing': False,
    'noise_while_growing_interval': 0.15,  # Required if noise_while_growing == True
    'pruning_flag': False,
    'pruning_threshold': 0.03,  # Required if pruning_flag == True
    'gca_hidden_size': 5,
    'rm_hidden_size': 5,
    'wp_hidden_size': 5,
    'graph_n_inputs': 2,
    'graph_n_outputs': 1,
    'hebbian': False
}
 


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Utilities
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def get_number_of_model_parameters(model:nn.Module):
    return sum(p.numel() for p in model.parameters())


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Developmental Program (Evolutionary-based NDP)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class NeuralDevelopmentalProgram:

    def __init__(self, config:dict = None):
        self._set_config(config)
        self._check_valid_config()

    def _set_default_config(self):
        self.config = dict(DEFAULT_PARAMETERS)

    def _set_config(self, config:dict):
        # Set default parameters
        self._set_default_config()
        if config is not None:
            # Set those values
            for key in self.config:
                if key in config:
                    self.config[key] = config[key]
                else:
                    # Showing if a variable was not defined
                    warnings.warn(f'Variable {key} not defined. Using default value {self.config[key]}.')
        # Set all variables
        self.state_dim = self.config['state_dim']
        self.weighted_graph_flag = self.config['weighted_graph_flag']
        self.initial_graph = self.config['initial_graph']
        self.network_extra_thinking = self.config['network_extra_thinking']
        self.initial_node_state_mode = self.config['initial_node_state_mode']
        self.shared_initial_node_state = self.config['shared_initial_node_state']
        self.add_hidden_node_to_minimal_network = self.config['add_hidden_node_to_minimal_network']
        self.pruning_flag = self.config['pruning_flag']
        self.pruning_threshold = self.config['pruning_threshold']
        self.graph_n_inputs = self.config['graph_n_inputs']
        self.graph_n_outputs = self.config['graph_n_outputs']
        self.noise_while_growing = self.config['noise_while_growing']
        self.noise_while_growing_interval = self.config['noise_while_growing_interval']
        # Create the MLPs
        self.graph_cellular_automata = GraphCellularAutomata(self.state_dim, self.config['gca_hidden_size'])
        self.replication_model = ReplicationModel(self.state_dim, self.config['rm_hidden_size'])
        self.weight_prediction_model = WeightPredictionModel(self.state_dim, self.config['wp_hidden_size'])
        # Check if config values from argument are valid

    def _check_valid_config(self):
        """
        This function checks that some of the input values for each variable is valid.
        FIX: I should do this for all the variables.
        Variables checked: 3/13
        """
        if self.state_dim < 1:
            raise ValueError('State dimension should be equal or greater than 1.')
        initial_graph_options = ['minimal_network', 'one_node']
        if self.initial_graph not in initial_graph_options:
            raise ValueError(f'Invalid value for the initial graph. Valid options are: {initial_graph_options}.')
        initial_node_state_mode_options = ['coevolve', 'ones', 'random', 'random_shared']
        if self.initial_node_state_mode not in initial_node_state_mode_options:
            raise ValueError(f'Invalid value for the initial node state mode. Valid options are: {initial_node_state_mode_options}.')
        if self.initial_node_state_mode == 'random_shared':
            if not isinstance(self.shared_initial_node_state, np.ndarray):
                raise ValueError(f'If initial_node_state_mode is set to random_shared, then shared_initial_node_state needs to be defined as a np.array([state_dim]) instead of {type(self.shared_initial_node_state), self.shared_initial_node_state}.')
            elif self.shared_initial_node_state.shape[1] != self.state_dim:
                print(self.shared_initial_node_state.shape[1])
                raise ValueError(f'shared_initial_node_state ({self.shared_initial_node_state.shape[0]}) must be an array with state_dim ({self.state_dim}) elements.')

    def get_total_number_of_mlp_parameters(self) -> int:
        n_params = get_number_of_model_parameters(self.graph_cellular_automata)
        n_params += get_number_of_model_parameters(self.replication_model)
        if self.weighted_graph_flag:
            n_params += get_number_of_model_parameters(self.weight_prediction_model)
        return n_params
    
    def update_mlp_weights(self, weights):
        """
        This function sets the weights of the MLPs.
        """
        models = [
            self.graph_cellular_automata,
            self.replication_model,
        ]
        if self.weighted_graph_flag:
            models.append(self.weight_prediction_model)

        if isinstance(weights, np.ndarray):
            weights = torch.tensor(weights, dtype=torch.float32)

        pointer = 0
        for model in models:
            for param in model.parameters():
                n_params = param.numel()
                new_values = weights[pointer:pointer + n_params]
                new_values = new_values.view_as(param)
                param.data.copy_(new_values)
                pointer += n_params

    def _genereate_node_state(self) -> np.array:
        """
        This function generates an array to initialise the state of a node. 
        This is mainly employed while initialising the graph.
        """
        if self.initial_node_state_mode in ['random', 'random_shared']:
            return np.random.uniform(-1, 1, size=(1, self.state_dim)).astype(np.float32)
        else:
            return np.ones((1, self.state_dim)).astype(np.float32)
    
    def generate_initial_seed_graph(self) -> Graphnx:
        """
        Generate an initial graph.
        There are two options available:
        'one_node' -> Creates a network with only one node
        'minimal_network' -> All inputs are connect to all outputs (a hidden node could be added too)
        """
        # Create graph
        graph = Graphnx(self.state_dim, self.weighted_graph_flag)

        # One node initial graph
        if self.initial_graph == 'one_node':
            if self.initial_node_state_mode in ['coevolve', 'random_shared']:
                node_state = self.shared_initial_node_state.copy()
            else:
                node_state = self._genereate_node_state()
            node_id = graph.add_node(node_state)
            graph.add_edge(node_id, node_id)

        # Minimal network (all inputs connected to outputs) 
        else:
            raise NotImplementedError("Come on, Daniel! You haven't implemented this yet! Stop procrastinating and just do it!")

        return graph

    
    def graph_convolution(self, graph:Graphnx, steps:int) -> Graphnx:
        """
        This perfroms the graph convolution. 
        The update is done by a Graph Cellular Automata.
        Pseudocode:
        For n in range(steps):
            new_states <- weights * states
            new_states <- graph_cellular_automata(new_states)
            states <- new_states
        """
        for _ in range(steps):
            weights = graph.get_adjacency_matrix()
            new_states = weights.T @ graph.nodes_states
            new_states = torch.tensor(new_states, dtype=torch.float32)
            new_states = self.graph_cellular_automata(new_states).numpy()
            graph.nodes_states = new_states.copy()
        return graph

    def grow_graph(self, graph:Graphnx) -> Graphnx:
        """
        Replication model R determines nodes in growing state
        New nodes are added to each of the growing nodes and their immediate neighbors
        New nodes' embeddings are defined as the mean embeddings of their parent nodes
        """
        # Use the Replication model to decide if a node should be replicad
        nodes_states = torch.tensor(graph.nodes_states)
        replicate_probabilities = self.replication_model(nodes_states).numpy()
        nodes_to_replicate = np.where(replicate_probabilities > 0)[0]
        all_neighbors = [graph.get_neighbors(node) for node in nodes_to_replicate]
        for i, node in enumerate(nodes_to_replicate):
            neighbors = all_neighbors[i]
            neighbors.add(node)
            neighbors_states = nodes_states[list(neighbors)]
            new_node_state = neighbors_states.mean(dim=0).numpy()
            if self.noise_while_growing:
                # Included a bit of noise so that the new nodes states to avoid states 
                # converging to a same value during the development process
                new_node_state += np.random.uniform(-self.noise_while_growing_interval, self.noise_while_growing_interval, new_node_state.size)
            new_node_id = graph.add_node(new_node_state)
            for neighbor in neighbors:
                # graph.add_edge(new_node_id, neighbor)
                graph.add_edge(neighbor, new_node_id)
        return graph
    
    def predict_weights(self, graph:Graphnx) -> Graphnx:
        """
        Weight update model W updates connectivity for each pair of nodes based on their concatenated embeddings.
        There are two versions:
        1. The first version only upudates existing edges, similar to the original implementation.
        2. The second verstion updates all possible pair of nodes. This matches more how it's described in the original paper. 
        Choosing version one, mainly because it makes everything faster.
        Fix: On version 2, two edges for each pair are created. But how do I choose which one to create when it doesn't exist?
        """
        # 1st version: only updates existing edges
        weights = graph.get_adjacency_matrix()
        for input_id, output_id in graph.edges():
            input_node_state = torch.tensor(graph.nodes_states[input_id], dtype=torch.float32)
            output_node_state = torch.tensor(graph.nodes_states[output_id], dtype=torch.float32)
            # print('input', input_node_state)
            # print('output', output_node_state)
            new_weight = self.weight_prediction_model(input_node_state, output_node_state).item()
            weights[input_id, output_id] = new_weight
        graph.update_adjacency_matrix(weights)

        # 2nd veresion: update values for all pair of nodes in the graph
        # for input_node in graph.nodes:
        #     for output_node in graph.nodes:
        #         if not graph.is_this_edge_valid(input_node, output_node):
        #             continue
        #         input_node_state = torch.tensor(input_node.state, dtype=torch.float32)
        #         output_node_state = torch.tensor(output_node.state, dtype=torch.float32)
        #         new_weight = self.weight_prediction_model(input_node_state, output_node_state).item()
        #         # graph.edges[(input_node.node_id, output_node.node_id)] = new_weight
        #         graph.add_edge(input_node.node_id, output_node.node_id, new_weight)

        return graph

    def prune(self, graph:Graphnx) -> Graphnx:
        """
        Edges with weights below pruning threshold P are removed.
        """
        # Find edges to remove
        weights = graph.get_adjacency_matrix()
        edges_to_remove = []
        for input_id, output_id in graph.edges():
            w = weights[input_id, output_id]
            if abs(w) < self.pruning_threshold:
                edges_to_remove.append((input_id, output_id))

        # Remove edges
        for input_id, output_id in edges_to_remove:
            graph.remove_edge(input_id, output_id)

            # Check that removed edge does not disconnect the outputs from the inputs
            """
            FIX: Should I check that the removed edge does not disconnect the outputs from the inputs?
            """

        return graph

    def _run_a_developmental_cycle(self, graph:Graphnx) -> Graphnx:
        """
        This method runs one developmental cycle.
        Steps are as follow:
        1. Compute graph diameter
        2. Graph convolution
        3. Grow graph
        4. Predict weights (if weighted graph)
        5. Prune (if option chosen)

        Modification: 
        I realised that I don't have to predict the weights in every cycle unless prunning is happening. 
        So, if pruning is not happening, the weights will be predicted only one time after the development is done. 

        Desmodification:
        I realised that I was wrong. The reason they need to predict the weights everytime is because
        the weights are used during the graph convolution. 
        """
        # Compute network diameter D
        diameter = graph.get_diameter()

        # Propagate nodes states En via graph convolution D steps
        steps = diameter + self.network_extra_thinking
        graph = self.graph_convolution(graph, steps)
        # graph.summary(full=True)
        
        # Replication model R determines nodes in growing state
        # New nodes are added to each of the growing nodes and their immediate neighbors
        # New nodes' embeddings are defined as the mean embeddings of their parent nodes
        graph = self.grow_graph(graph)
        # graph.summary(full=True)

        # If graph is weighted then
        if self.weighted_graph_flag:
            # Weight update model W updates connectivity for each pair of nodes based on their concatenated embeddings
            graph = self.predict_weights(graph)
        
        # If pruning then
        if self.pruning_flag:
            # Edges with weights below pruning threshold P are removed
            graph = self.prune(graph)

        return graph

    def develope(self, n_cycles:int, debug:bool=False) -> Graphnx:
        """
        This function developes a graph from scratch for a defined number of cycles.
        """
        start_time = time.time()
        with torch.no_grad():
            graph = self.generate_initial_seed_graph()
            if debug:
                print('Initial graph')
                graph.summary(full=False)
            for i in range(n_cycles):
                graph = self._run_a_developmental_cycle(graph)
                if debug:
                    print(f'Graph at cycle {i}')
                    graph.summary(full=False)
    
        if debug:
            print(f'Total development time = {time.time() - start_time}')
        return graph

    def summary(self):
        print('-------------------------------------')
        print('NDP')
        print('-------------------------------------')
        print('Graph')
        print(f'Node internal state size = {self.state_dim}')
        print(f'Number of inputs = {self.graph_n_inputs}')
        print(f'Number of outputs = {self.graph_n_outputs}')
        print('-------------------------------------')
        print('MLP Parameters')
        print(f'Graph Cellular Automata = {get_number_of_model_parameters(self.graph_cellular_automata)}')
        print(f'Replication Model = {get_number_of_model_parameters(self.replication_model)}')
        print(f'Weight Prediction Model = {get_number_of_model_parameters(self.weight_prediction_model)}')
        print(f'Total = {self.get_total_number_of_mlp_parameters()}')
        print('-------------------------------------\n')



'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function (mainly for testing)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':

    ndp = NeuralDevelopmentalProgram()
    graph = ndp.develope(n_cycles=5, debug=True)
    print(ndp.get_total_number_of_mlp_parameters())
    # graph.summary()
