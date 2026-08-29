'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import jax
import jax.numpy as jnp
import time
import warnings
from dataclasses import replace

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from NDP.mlps_jax import GraphCellularAutomata, ReplicationModel, WeightPredictionModel
from Graph.graph_jax import GraphJax

# from Utilities.utilities import get_number_of_model_parameters

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
    'hebbian': False,
    'model': 'standard_ndp'
}
 

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Developmental Program (Evolutionary-based NDP)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class NeuralDevelopmentalProgramJax:

    # ---------------------------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------------------------

    def __init__(self, config:dict = None):
        # Set the algorithm configuration
        self._set_config(config)
        # Check if config values from argument are valid
        self._check_valid_config()

    def _set_default_config(self, default_parameters=DEFAULT_PARAMETERS):
        self.config = dict(default_parameters)
    
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
        self._set_all_variables()
        # Create the MLPs
        self._initialise_mlps()

    def _set_all_variables(self):
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
        
    def _initialise_mlps(self):
        self.graph_cellular_automata = GraphCellularAutomata(self.state_dim, self.config['gca_hidden_size'])
        self.replication_model = ReplicationModel(self.state_dim, self.config['rm_hidden_size'])
        self.weight_prediction_model = WeightPredictionModel(self.state_dim, self.config['wp_hidden_size'])

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
        # if self.initial_node_state_mode == 'random_shared':
        #     if not isinstance(self.shared_initial_node_state, np.ndarray):
        #         raise ValueError(f'If initial_node_state_mode is set to random_shared, then shared_initial_node_state needs to be defined as a np.array([state_dim]) instead of {type(self.shared_initial_node_state), self.shared_initial_node_state}.')
        #     elif self.shared_initial_node_state.shape[1] != self.state_dim:
        #         print(self.shared_initial_node_state.shape[1])
        #         raise ValueError(f'shared_initial_node_state ({self.shared_initial_node_state.shape[0]}) must be an array with state_dim ({self.state_dim}) elements.')

    # ---------------------------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------------------------

    def get_total_number_of_mlp_parameters(self) -> int:
        n_params = [model.get_n_parameters() for model in self._get_mlp_models()]
        return sum(n_params)
    
    # ---------------------------------------------------------------------------------------
    # MLPs
    # ---------------------------------------------------------------------------------------

    def _get_mlp_models(self) -> list:
        models = [
            self.graph_cellular_automata,
            self.replication_model,
        ]
        if self.weighted_graph_flag:
            models.append(self.weight_prediction_model)
        return models

    
    def unpack_mlp_parameters(self, weights):

        weights = jnp.asarray(weights, dtype=jnp.float32)

        pointer = 0
        params = {}

        # Graph Cellular Automata
        n_params = self.graph_cellular_automata.get_n_parameters()
        params["gca"], used = self.graph_cellular_automata.unpack_parameters(weights[pointer:pointer + n_params])
        pointer += used

        # Replication model
        n_params = self.replication_model.get_n_parameters()
        params["replication"], used = self.replication_model.unpack_parameters(weights[pointer:pointer + n_params])
        pointer += used

        # Weight prediction model
        if self.weighted_graph_flag:
            n_params = self.weight_prediction_model.get_n_parameters()
            params["weight_prediction"], used = self.weight_prediction_model.unpack_parameters(weights[pointer:pointer + n_params])
            pointer += used
        return params

    # ---------------------------------------------------------------------------------------
    # Developmental Process
    # ---------------------------------------------------------------------------------------

    def _genereate_node_state(self, key) -> jax.Array:
        """
        This function generates an array to initialise the state of a node. 
        This is mainly employed while initialising the graph.
        """
        if self.initial_node_state_mode in ['random', 'random_shared']:
            return jax.random.uniform(key, shape=(self.state_dim,), minval=-1.0, maxval=1.0, dtype=jnp.float32)
        else:
            return jnp.ones(self.state_dim, dtype= jnp.float32)
    
    def generate_initial_seed_graph(self, key) -> GraphJax:
        """
        Generate an initial graph.
        There are two options available:
        'one_node' -> Creates a network with only one node
        'minimal_network' -> All inputs are connect to all outputs (a hidden node could be added too)
        """
        # Create graph
        graph = GraphJax.create(max_nodes=self.max_nodes, state_dim=self.state_dim, weighted_graph_flag=self.weighted_graph_flag)

        # One node initial graph
        if self.initial_graph == 'one_node':
            if self.initial_node_state_mode in ['coevolve', 'random_shared']:
                node_state = jnp.asarray(self.shared_initial_node_state, dtype=jnp.float32).reshape(self.state_dim)
            else:
                node_state = self._genereate_node_state(key)
            graph, node_id = graph.add_node(node_state)
            graph = graph.add_edge(node_id, node_id)

        # Minimal network (all inputs connected to outputs) 
        else:
            raise NotImplementedError("Come on, Daniel! You haven't implemented this yet! Stop procrastinating and just do it!")

        return graph

    def graph_convolution(self, graph:GraphJax, steps:int, params:dict) -> GraphJax:
        """
        This perfroms the graph convolution. 
        The update is done by a Graph Cellular Automata.
        Pseudocode:
        For n in range(steps):
            new_states <- weights * states
            new_states <- graph_cellular_automata(new_states)
            states <- new_states
        """
        def convolution_step(_, graph):
            weights = graph.weights
            new_states = weights.T @ graph.nodes_states
            new_states = self.graph_cellular_automata.forward(new_states, params['gca'])
            return replace(graph, nodes_states=new_states)
        graph = jax.lax.fori_loop(0, steps, convolution_step, graph)
        return graph

    def grow_graph(self, graph:GraphJax, params:dict, key) -> GraphJax:
        """
        Replication model R determines nodes in growing state
        New nodes are added to each of the growing nodes and their immediate neighbors
        New nodes' embeddings are defined as the mean embeddings of their parent nodes
        """
        # Use the Replication model to decide if a node should be replicated
        replication_values = self.replication_model.forward(graph.nodes_states, params['replication'])[..., 0]
        replicate_mask = (replication_values > 0) & graph.node_mask

        # Assign new nodes IDs
        n_active = graph.number_of_nodes()
        replication_rank = jnp.cumsum(replicate_mask.astype(jnp.int32)) - 1
        new_node_ids = n_active + replication_rank
        replica_assignment = jax.nn.one_hot(new_node_ids, graph.max_nodes, dtype=jnp.float32)
        replica_assignment *= replicate_mask[:, None]

        # Get the neighborhood of every node
        neighborhood = graph.get_neighbor_matrix(include_self_node=True)
        neighborhood = neighborhood.astype(jnp.float32)
        neighborhood_size = jnp.sum(neighborhood, axis=1, keepdims=1)
        new_states = (neighborhood @ graph.nodes_states) / jnp.maximum(neighborhood_size, 1.0)

        # Add noise while growing if option was selected
        if self.noise_while_growing:
            noise = jax.random.uniform(key, shape=new_states.shape, minval=-self.noise_while_growing_interval, maxval=self.noise_while_growing_interval)
            new_states = new_states + noise

        # Add nodes
        new_state_slots = replica_assignment.T @ new_states
        new_nodes_mask = jnp.any(replica_assignment > 0, axis=0)
        nodes_states = jnp.where(new_nodes_mask[:, None], new_state_slots, graph.nodes_states)
        node_mask = graph.node_mask | new_nodes_mask

        # Add edges
        new_edges = (neighborhood.T @ replica_assignment) > 0
        adjacency = graph.adjacency | new_edges
        weights = jnp.where(new_edges, 1.0, graph.weights)

        return replace(graph, nodes_states=nodes_states, node_mask=node_mask, adjacency=adjacency, weights=weights)

    
    def predict_weights(self, graph:GraphJax, params=dict) -> GraphJax:
        """
        Weight update model W updates connectivity for each pair of nodes based on their concatenated embeddings.
        """
        source_states = graph.nodes_states[:, None, :]
        target_states = graph.nodes_states[None, :, :]

        source_states = jnp.broadcast_to(source_states, (graph.max_nodes, graph.max_nodes, self.state_dim))
        target_states = jnp.broadcast_to(target_states, (graph.max_nodes, graph.max_nodes, self.state_dim))

        new_weights = self.weight_prediction_model.forward(source_states, target_states, params['weight_prediction'])
        new_weights = new_weights[..., 0]
        new_weights = jnp.where(graph.adjacency, new_weights, 0.0)

        return replace(graph, weights=new_weights)

    def prune(self, graph:GraphJax) -> GraphJax:
        """
        Edges with weights below pruning threshold P are removed.
        """
        keep_edges = jnp.abs(graph.weights) >= self.pruning_threshold
        adjacency = graph.adjacency & keep_edges
        weights = jnp.where(adjacency, graph.weights, 0.0)
        return replace(graph, adjacency=adjacency, weights=weights)

    def _run_a_developmental_cycle(self, graph:GraphJax, params:dict, key) -> GraphJax:
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
        # diameter = graph.get_diameter()
        diameter = graph.get_largest_subgraph_diameter()

        # Propagate nodes states En via graph convolution D steps
        steps = diameter + self.network_extra_thinking
        graph = self.graph_convolution(graph, steps, params)
        # graph.summary(full=True)
        
        # Replication model R determines nodes in growing state
        # New nodes are added to each of the growing nodes and their immediate neighbors
        # New nodes' embeddings are defined as the mean embeddings of their parent nodes
        graph = self.grow_graph(graph, params, key)
        # graph.summary(full=True)

        # If graph is weighted then
        if self.weighted_graph_flag:
            # Weight update model W updates connectivity for each pair of nodes based on their concatenated embeddings
            graph = self.predict_weights(graph, params)
        
        # If pruning then
        if self.pruning_flag:
            # Edges with weights below pruning threshold P are removed
            graph = self.prune(graph)

        return graph

    def develope(self, n_cycles:int, params:dict, key, debug:bool=False) -> GraphJax:
        """
        This function developes a graph from scratch for a defined number of cycles.
        """
        start_time = time.time()
        self.max_nodes = 2 ** n_cycles

        keys = jax.random.split(key, n_cycles + 1)

        graph = self.generate_initial_seed_graph(keys[0])
        if debug:
            print('Initial graph')
            graph.summary(full=False)
        for i in range(n_cycles):
            graph = self._run_a_developmental_cycle(graph, params, key[i+1])
            if debug:
                print(f'Graph at cycle {i}')
                graph.summary(full=False)
    
        if debug:
            print(f'Total development time = {time.time() - start_time}')
        return graph

    # ---------------------------------------------------------------------------------------
    # Summary 
    # ---------------------------------------------------------------------------------------

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
        print(f'Graph Cellular Automata = {self.graph_cellular_automata.get_n_parameters()}')
        print(f'Replication Model = {self.replication_model.get_n_parameters()}')
        print(f'Weight Prediction Model = {self.weight_prediction_model.get_n_parameters()}')
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
