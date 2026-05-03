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
from Graph.ndp_graph import Node, Graph

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Default parameters
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

NDP_PARAMETERS = {
    'state_dim' : 5,
    'weighted_graph_flag' : True,
    'pruning_flag' : False,
    'pruning_threshold': 0.01,
    'initialise_graph_w_hidden_node_flag' : True,
    'gca_hidden_size' : 5,
    'rm_hidden_size' : 5,
    'wp_hidden_size' : 5
}

TASK = {
    'n_inputs' : 2,
    'n_outputs' : 1
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

    def __init__(self, config = None):
        self._set_config(config)

    def _set_default_config(self):
        self.config = {
            'state_dim' : 5,
            'weighted_graph_flag' : True,
            'initial_graph' : 'minimal_network',
            'node_state_random_init' : False,
            'add_hidden_node_to_minimal_network' : True,
            'pruning_flag' : False,
            'pruning_threshold': 0.01,
            'gca_hidden_size' : 5,
            'rm_hidden_size' : 5,
            'wp_hidden_size' : 5,
            'graph_n_inputs' : 2,
            'graph_n_outputs' : 1
        }

    def _set_config(self, config):
        # Set default parameters
        self._set_default_config()
        if config is not None:
            # Set those values
            for key in config:
                self.config[key] = config[key]
            # Showing if a variable was not defined
            for key in self.config:
                if key not in config:
                    warnings.warn(f'Variable {key} not defined. Using default value {self.config[key]}.')
            # Check if config values from argument are valid
            self._check_valid_config()
        # Create the MLPs
        self.graph_cellular_automata = GraphCellularAutomata(self.config['state_dim'], self.config['gca_hidden_size'])
        self.replication_model = ReplicationModel(self.config['state_dim'], self.config['rm_hidden_size'])
        self.weight_prediction_model = WeightPredictionModel(self.config['state_dim'], self.config['wp_hidden_size'])

    def _check_valid_config(self):
        """
        This function checks that some of the input values for each variable is valid.
        FIX: I should do this for all the variables.
        """
        if self.config['state_dim'] < 1:
            raise ValueError('State dimension should be equal or greater than 1.')
        initial_graph_options = ['minimal_network', 'one_node']
        if self.config['initial_graph'] not in initial_graph_options:
            raise ValueError(f'Invalid value for the initial graph. Valid options are: {initial_graph_options}')

    def get_total_number_of_mlp_parameters(self):
        n_params = get_number_of_model_parameters(self.graph_cellular_automata)
        n_params += get_number_of_model_parameters(self.replication_model)
        if self.config['weighted_graph_flag']:
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
        if self.config['weighted_graph_flag']:
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

        if self.config['node_state_random_init']:
            return np.random.uniform(-1, 1, size=(1, self.config['state_dim'])).astype(np.float32)
        else:
            return np.ones((1, self.config['state_dim'])).astype(np.float32)
    
    def generate_initial_seed_graph(self):
        """
        Generate an initial graph.
        There are two options available:
        'one_node' -> Creates a network with only one node
        'minimal_network' -> All inputs are connect to all outputs (a hidden node could be added too)
        """

        # Create graph
        graph = Graph(self.config['weighted_graph_flag'])

        # One node initial graph
        if self.config['initial_graph'] == 'one_node':
            node = Node(state_dim=self.config['state_dim'], node_type='input')
            node.state = self._genereate_node_state()
            graph.add_node(node)

        # Minimal network (all inputs connected to outputs) 
        else:  
            # Add input nodes
            for _ in range(self.config['graph_n_inputs']):
                node = Node(state_dim=self.config['state_dim'], node_type='input')
                node.state = self._genereate_node_state()
                graph.add_node(node)

            # Add output nodes
            for _ in range(self.config['graph_n_outputs']):
                node = Node(state_dim=self.config['state_dim'], node_type='output')
                node.state = self._genereate_node_state()
                graph.add_node(node)
                if not self.config['initial_graph']:
                    for i in range(self.config['graph_n_inputs']):
                        graph.add_edge(i, len(graph.nodes) - 1)

            # Add hidden node (if option was selected)
            if self.config['add_hidden_node_to_minimal_network']:
                node = Node(state_dim=self.config['state_dim'], node_type='hidden')
                node.state = self._genereate_node_state()
                graph.add_node(node)
                for i in range(self.config['graph_n_inputs']):
                        graph.add_edge(i, len(graph.nodes) - 1)
                for i in range(self.config['graph_n_outputs']):
                        graph.add_edge(len(graph.nodes) - 1, self.config['graph_n_inputs'] + i)

        return graph

    def graph_convolution(self, graph:Graph, steps:int) -> Graph:
        """
        This perfroms the graph convolution. 
        The update is done by a Graph Cellular Automata.
        """

        for _ in range(steps):
            updated_nodes = list(graph.nodes)

            for i, node in enumerate(graph.nodes):
                _, neighbors_states = graph.get_neighbors_states(node.node_id)
                node_state = torch.tensor(node.state)
                neighbors_states = torch.tensor(neighbors_states)
                if neighbors_states.size()[0] == 0:
                    neighbors_states = node_state
                updated_nodes[i].state = self.graph_cellular_automata(node_state, neighbors_states).numpy()
            
            graph.nodes = list(updated_nodes)
        
        return graph

    def _set_node_type(self, node_id:int) -> str:
        """
        This sets the type of each node.
        The first nodes will always be inputs, followed by the outputs and every node after that will be hidden nodes.
        """

        if node_id < self.config['graph_n_inputs']:
            return 'input'
        elif node_id < self.config['graph_n_inputs'] + self.config['graph_n_outputs']:
            return 'output'
        else:
            return 'hidden'
 
    def grow_graph(self, graph:Graph) -> Graph:
        """
        Replication model R determines nodes in growing state
        New nodes are added to each of the growing nodes and their immediate neighbors
        New nodes' embeddings are defined as the mean embeddings of their parent nodes
        """
        
        new_nodes = [] 
        new_edges = {}
        for node in graph.nodes:
            # Use the Replication model to decide if a node should be replicated
            state = torch.tensor(node.state, dtype=torch.float32)
            replicate_node = self.replication_model(state)
            # print(replicate_node)
            # In case it does:
            if replicate_node:
                # Create a new node
                new_node_id = len(graph.nodes) + len(new_nodes)
                new_node_type = self._set_node_type(new_node_id)
                new_node = Node(node_id=new_node_id, state_dim=self.config['state_dim'], node_type=new_node_type)
                # Compute the mean of the neighbors
                neighbor_ids, neighbors_states = graph.get_neighbors_states(node.node_id)
                # neighbors_states = torch.tensor(neighbors_states)
                if len(neighbors_states) == 0:
                    new_node.state = node.state.copy()
                else:
                    neighbors_states = np.vstack([neighbors_states, node.state])
                    mean_state = np.mean(neighbors_states, axis = 0)
                    new_node.state = np.expand_dims(mean_state, axis=0)
                new_nodes.append(new_node)
                # Add the edges to the new node
                """
                FIX: I set that the outputs will always be on the end of each edge. 
                Is this okay tho?
                """
                for id in neighbor_ids + [node.node_id]:
                    neighbor_type = graph.nodes[id].node_type
                    if neighbor_type == 'output':
                        new_edges[(new_node_id, id)] = 1.0
                    else:
                        new_edges[(id, new_node_id)] = 1.0

        # Add the new nodes and edges to the graph
        graph.nodes.extend(new_nodes)
        # graph.edges |= new_edges
        for (input_node, output_node), weight in new_edges.items():
            graph.add_edge(input_node, output_node, weight)
        return graph
    
    def predict_weights(self, graph:Graph) -> Graph:
        """
        Weight update model W updates connectivity for each pair of nodes based on their concatenated embeddings.
        There are two versions:
        1. The first version only upudates existing edges, similar to the original implementation.
        2. The second verstion updates all possible pair of nodes. This matches more how it's described in the original paper. 
        Chossing version one, mainly because it makes everything faster.
        Fix: On version 2, two edges for each pair are created. But how do I choose which one to create when it doesn't exist?
        """

        # 1st version: only updates existing edges
        for input_id, output_id in graph.edges.keys():
            input_node, output_node = graph.get_multiple_nodes([input_id, output_id])
            input_node_state = torch.tensor(input_node.state, dtype=torch.float32)
            output_node_state = torch.tensor(output_node.state, dtype=torch.float32)
            new_weight = self.weight_prediction_model(input_node_state, output_node_state).item()
            graph.edges[(input_id, output_id)] = new_weight

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

    def prune(self, graph:Graph) -> Graph:
        """
        Edges with weights below pruning threshold P are removed.
        """
        
        # Find edges to remove
        edges_to_remove = []
        for edge, weight in graph.edges.items():
            if abs(weight) < self.config['pruning_threshold']:
                edges_to_remove.append(edge)

        # Remove edges
        for edge in edges_to_remove:
            (input_node, output_node) = edge
            weight = graph.edges[edge]
            graph.delete_edge(input_node, output_node)
            # del graph.edges[edge]

            # Check that removed edge does not disconnect the outputs from the inputs
            if not graph.is_there_a_path_to_the_output():
                # graph.edges[edge] = weight
                graph.add_edge(input_node, output_node, weight)

        return graph

    def run_a_developmental_cycle(self, graph:Graph, debug:bool=True) -> Graph:
        """
        This method runs one developmental cycle.
        Steps are as follow:
        1. Compute graph diameter
        2. Graph convolution
        3. Grow graph
        4. Predict weights (if weighted graph)
        5. Prune (if option chosen)
        """

        # Compute network diameter D
        if debug:
            start_time = time.time()
            print(f'A - n_nodes = {len(graph.nodes)}')

        diameter = graph.get_diameter()
        # diameter = int(max(1, len(graph.nodes)))

        if debug:
            print(f'Time = {time.time() - start_time}')
            start_time = time.time()
            # print(diameter)

            print('B')
        # Propagate nodes states En via graph convolution D steps
        graph = self.graph_convolution(graph, diameter)
        
        if debug:
            print(f'Time = {time.time() - start_time}')
            start_time = time.time()
            # graph.summary()

            print('C')
        # Replication model R determines nodes in growing state
        # New nodes are added to each of the growing nodes and their immediate neighbors
        # New nodes' embeddings are defined as the mean embeddings of their parent nodes
        graph = self.grow_graph(graph)
        
        if debug:
            print(f'Time = {time.time() - start_time}')
            start_time = time.time()
            # graph.summary()

            print(f'D - n_nodes = {len(graph.nodes)}')
        # If weighted network then:
        if self.config['weighted_graph_flag']:
            # Weight update model W updates connectivity for each pair of nodes based on their concatenated embeddings
            graph = self.predict_weights(graph)
        if debug:
            print(f'Time = {time.time() - start_time}')
            start_time = time.time()
            # graph.summary()

            print('E')
        # If pruning then
        if self.config['pruning_flag']:
            # Edges with weights below pruning threshold P are removed
            graph = self.prune(graph)
        if debug:
            print(f'Time = {time.time() - start_time}')
        return graph

    def develope(self, n_cycles:int, debug:bool=False) -> Graph:
        """
        This function developes a graph from scratch for a defined number of cycles.
        """

        with torch.no_grad():
            graph = self.generate_initial_seed_graph()
            # print('Initial graph')
            # graph.summary()
            # print(len(graph.nodes))
            for i in range(n_cycles):
                if debug:
                    print(f'Graph at cycle {i}')
                graph = self.run_a_developmental_cycle(graph, debug)
                # graph.summary()
            return graph

    def summary(self):
        print('-------------------------------------')
        print('NDP')
        print('-------------------------------------')
        print(f'Graph Cellular Automata = {get_number_of_model_parameters(self.graph_cellular_automata)}')
        print(f'Replication Model = {get_number_of_model_parameters(self.replication_model)}')
        print(f'Weight Prediction Model = {get_number_of_model_parameters(self.weight_prediction_model)}')
        print(f'Total = {self.get_total_number_of_mlp_parameters()}')
        print('-------------------------------------')



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
