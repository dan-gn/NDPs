'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch
from itertools import groupby

from Graph.graph import Graphnx
from NDP.ndp_nx import NeuralDevelopmentalProgram
from NDP.ndp_mlps import GraphCellularAutomata, CreateEdgeModel, RemoveEdgeModel

from Utilities.utilities import get_number_of_model_parameters

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Default parameters
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

DEFAULT_PARAMETERS = {
    'state_dim': 5,
    'n_nodes': 16,
    'initial_graph_density': 0.2,
    'weighted_graph_flag': True,
    'network_extra_thinking': 0,
    'initial_node_state_mode': 'coevolve', 
    'shared_initial_node_state': None,
    'noise_while_growing': False,
    'noise_while_growing_interval': 0.15,  # Required if noise_while_growing == True
    'pruning_flag': False,
    'gca_hidden_size': 5,
    'create_edge_hidden_size': 5,
    'remove_edge_hidden_size': 5,
    'graph_n_inputs': 2,
    'graph_n_outputs': 1,
    'hebbian': False,
    'model': 'hebbian_ndp'
}

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Developmental Program (Evolutionary-based NDP)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class HebbianNeuralDevelopmentalProgram(NeuralDevelopmentalProgram):

    # Init the same as the standart version
    def __init__(self, config:dict = None):
        super().__init__(config)

    # Redefine to update the default parameters
    def _set_default_config(self, default_parameters=DEFAULT_PARAMETERS):
        return super()._set_default_config(default_parameters)

    # Set all variables 
    def _set_all_variables(self):
        self.n_nodes = self.config['n_nodes']
        self.initial_graph_density = self.config['initial_graph_density']
        self.state_dim = self.config['state_dim']
        self.weighted_graph_flag = self.config['weighted_graph_flag']
        self.network_extra_thinking = self.config['network_extra_thinking']
        self.initial_node_state_mode = self.config['initial_node_state_mode']
        self.shared_initial_node_state = self.config['shared_initial_node_state']
        self.pruning_flag = self.config['pruning_flag']
        self.graph_n_inputs = self.config['graph_n_inputs']
        self.graph_n_outputs = self.config['graph_n_outputs']
        self.noise_while_growing = self.config['noise_while_growing']
        self.noise_while_growing_interval = self.config['noise_while_growing_interval']

    # Check valid configuration
    def _check_valid_config(self):
        if self.state_dim < 1:
            raise ValueError('State dimension should be equal or greater than 1.')
        if self.graph_n_inputs + self.graph_n_outputs > self.n_nodes:
            raise ValueError('The number of inputs plus outputs exceed the total number of nodes.')

    # Initialise MLPs for neural development process 
    def _initialise_mlps(self):
        self.graph_cellular_automata = GraphCellularAutomata(self.state_dim, self.config['gca_hidden_size'])
        self.create_edge_model = CreateEdgeModel(self.state_dim, self.config['create_edge_hidden_size'])
        self.remove_edge_model = RemoveEdgeModel(self.state_dim, self.config['remove_edge_hidden_size'])

    # Get MLP models
    def _get_mlp_models(self) -> list:
        models = [
            self.graph_cellular_automata,
            self.create_edge_model,
            self.remove_edge_model
        ]
        return models

    # Generates an initial grpah
    def generate_initial_seed_graph(self) -> Graphnx:
        """
        Generates an initial graph.
        The graph has N nodes (I inputs, H hidden and O outputs).
        The network consists on:
        1. All inputs connected to all outputs.
        2. Sparse edges between inputs to hidden nodes.
        3. Spare edges between hidden to hidden nodes.
        4. Spare edges between hidden to output nodes.
        """
        # Create graph
        graph = Graphnx(self.state_dim, self.weighted_graph_flag)

        # Add nodes
        # For coevolve the initial state of the graph comes from the vector that the evolutionary algorithm optimise
        if self.initial_node_state_mode == 'coevolve':
            coevolved_array = self.shared_initial_node_state.copy()
            # The first element is used as the seed to create the edges
            network_seed = int(np.clip(coevolved_array[0, 0] + 1, 0, 2) * 50)
            # The rest of the elements are used as the initial state of each node
            nodes_states = coevolved_array[0, 1:]
            nodes_states = nodes_states.reshape(self.n_nodes, self.state_dim)
            graph.add_nodes_from(nodes_states)

        # If random_shared, an initial state is generated ramdomly as it's used for all individuals during optimisation
        elif self.initial_node_state_mode == 'random_shared':
            raise NotImplementedError("Come on, Daniel! You haven't implemented this yet! Stop procrastinating and just do it!")

        # The other options are generate a new random or constant (ones) initial state
        else:
            for _ in range(self.n_nodes):
                node_state = self._genereate_node_state()
                graph.add_node(node_state)
            network_seed = np.random.randint()

        # Get list of input, hidden and output nodes
        input_nodes = np.arange(self.graph_n_inputs)
        hidden_nodes = np.arange(self.graph_n_inputs, self.n_nodes, self.graph_n_outputs)
        output_nodes = np.arange(self.n_nodes - self.graph_n_outputs, self.n_nodes)

        # Set seed to random number generator
        self.rng = np.random.default_rng(network_seed)
        density = self.initial_graph_density

        # Add edges from inputs to outputs nodes (Fully connected) 
        graph.add_sparse_edges(input_nodes, output_nodes, density=1.0, rng=self.rng)

        # Add edges from inputs to hidden nodes (sparse)
        graph.add_sparse_edges(input_nodes, hidden_nodes, density=density, rng=self.rng)

        # Add edges from hidden to hidden nodes (sparse)
        graph.add_sparse_edges(hidden_nodes, hidden_nodes, density=density, rng=self.rng)

        # Add edges from hidden to output nodes (sparse)
        graph.add_sparse_edges(hidden_nodes, output_nodes, density=density, rng=self.rng)

        return graph

    # Returns all possible edges from disconnected nodes in the graph 
    def get_all_disconnected_nodes(self, graph:Graphnx, allow_self_loops:bool=False) -> list:
        nodes = graph.nodes()
        disconnected = []
        for source in nodes:
            for target in nodes:
                # Ignore edges directed to input nodes
                if target < self.graph_n_inputs:
                    continue
                # Ignore self loops
                if not allow_self_loops and source == target:
                    continue
                # Check if edge doesn't exist
                if not graph.has_edge(source, target):
                    disconnected.append((source, target))
        return disconnected

    # Get disconnected nodes that are close to each other
    # This means that they are "two hops" distance (neighbour of neighbor nodes)    
    def get_two_hop_disconnected_nodes(self, graph:Graphnx, allow_self_loops:bool=False) -> list:
        nodes = graph.nodes()
        candidates = set()
        for source in nodes:
            one_hop = graph.successors(source)
            two_hop = set()
            for neighbour in one_hop:
                two_hop.update(graph.successors(neighbour))
            for target in two_hop:
                if not allow_self_loops and source == target:
                    continue
                if not graph.has_edge(source, target):
                    candidates.add((source, target))
        return list(candidates)

    # Sample n edges per source node 
    def sample_n_disconnected_edges_per_node(self, candidate_edges:list, candidates_per_node:int=2, allow_self_loops:bool=False) -> list:
        # Split by source
        rearranged_candidate_edges = [list(group) for source, group in groupby(candidate_edges, key=lambda x:x[0])] 

        # Sample the possible edges
        sampled_candidate_edges = []
        for possible_edges_per_source in rearranged_candidate_edges:
            if candidates_per_node < len(possible_edges_per_source):
                candidate_edges_per_source = self.rng.choice(possible_edges_per_source, size=candidates_per_node, replace=False)
                sampled_candidate_edges.extend(candidate_edges_per_source)
            else:
                sampled_candidate_edges.extend(possible_edges_per_source)

        return sampled_candidate_edges


        
    # Creates new edges using the MLP 
    def create_edges(self, graph:Graphnx) -> list:
        # Get all possible candidate edges
        option = 0
        if option == 0:
            candidate_edges = self.get_all_disconnected_nodes(graph)
        elif option == 1:
            candidate_edges = self.get_two_hop_disconnected_nodes(graph)

        # Sample to reduce computational cost
        candidate_edges = self.sample_n_disconnected_edges_per_node(candidate_edges)

        # Concatenate the source and target states
        node_states = np.asarray(graph.nodes_states)
        candidate_source, candidate_target = candidate_edges[:, 0], candidate_edges[:, 1]
        source_states, target_states = node_states[candidate_source], node_states[candidate_target]
        pair_source_target_states = np.concat((source_states, target_states))
        pair_source_target_states = torch.tensor(pair_source_target_states)

        # Use create_edge_model to decide if a candidate edge should be created
        model_decision = self.create_edge_model(pair_source_target_states).numpy()
        chosen_edges = model_decision > 0    # Threshold is 0 since the model uses a tanh
        edges_to_create = candidate_edges[chosen_edges]

        return edges_to_create

    # Removes existing edges using the MLP
    def remove_edges(self, graph:Graphnx) -> list:
        return []

    # Structural synapsis: create and remove edges
    def structural_synapsis(self, graph:Graphnx) -> Graphnx:
        """
        Here we are gonna modify the ANN structure. We should be able to:
        1. Add new edges - (Randomly sample possible new edges)
        2. Prune edges - (Check all edges)
        3. Split edges and adding new neurons - (Should I do this?)
        """
        # Figure out wich edges to add and which ones to remove
        edges_to_add = self.create_edges(graph)
        edges_to_remove = self.remove_edges(graph)

        # Add and remove edges
        graph.add_edges_from(edges_to_add)
        graph.remove_edges_from(edges_to_remove)

        return graph

    # Developmental cycle 
    def _run_a_developmental_cycle(self, graph:Graphnx) -> Graphnx:
        """
        This method runs one developmental cycle.
        Steps are as follow:
        1. Compute graph diameter
        2. Graph convolution
        3. Structural synapsis
        """
        # Compute network diameter D
        diameter = graph.get_largest_subgraph_diameter()

        # Propagate nodes states En via graph convolution D steps
        steps = diameter + self.network_extra_thinking
        graph = self.graph_convolution(graph, steps)

        # Structural Synapsis (Add and remove edges)
        graph = self.structural_synapsis(graph)

        return graph

    # Print NDP Summary
    def summary(self):
        print('-------------------------------------')
        print('Variant NDP')
        print('-------------------------------------')
        print('Graph')
        print(f'Node internal state size = {self.state_dim}')
        print(f'Number of inputs = {self.graph_n_inputs}')
        print(f'Number of outputs = {self.graph_n_outputs}')
        print('-------------------------------------')
        print('MLP Parameters')
        print(f'Graph Cellular Automata = {get_number_of_model_parameters(self.graph_cellular_automata)}')
        print(f'Create Edge Model = {get_number_of_model_parameters(self.create_edge_model)}')
        print(f'Remove Edges Model = {get_number_of_model_parameters(self.remove_edge_model)}')
        print(f'Total = {self.get_total_number_of_mlp_parameters()}')
        print('-------------------------------------\n')

