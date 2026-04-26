'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

import numpy as np
import torch 
import torch.nn as nn

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Graph.ndp_graph import Node, Graph

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Default parameters
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

NDP_PARAMETERS = {
    'state_dim' : 2,
    'weighted_graph_flag' : True,
    'pruning_flag' : True,
    'pruning_threshold': 0.01,
    'initialise_graph_w_hidden_node_flag' : True,
    'gca_hidden_size' : 32,
    'rm_hidden_size' : 32,
    'wp_hidden_size' : 32
}

TASK = {
    'n_inputs' : 1,
    'n_outputs' : 1
}

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Utilities
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def xor(x):
    pass


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
            nn.Linear(state_dim * 2, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, state_dim)
        )

    def forward(self, node_state, neighbors_states):
        if neighbors_states.size()[0] > 1:
            neighbors_states_mean = torch.mean(neighbors_states, axis = 0).unsqueeze(axis = 0)
            x = torch.cat([node_state, neighbors_states_mean], dim = 1)
        else:
            x = torch.cat([node_state, neighbors_states], dim = 1)

        delta = self.model(x)
        new_state = node_state + delta

        return new_state

# Replication model employed to grow the graph 
class ReplicationModel(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
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
        x = torch.cat([input_node_state, output_node_state], dim=1)
        return self.model(x)


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Neural Developmental Program (Evolutionary-based NDP)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class NeuralDevelopmentalProgram:

    def __init__(self, ndp_parameters=NDP_PARAMETERS, task=TASK):
        # Flags
        self.weighted_graph_flag = ndp_parameters['weighted_graph_flag']
        self.pruning_flag = ndp_parameters['pruning_flag']
        self.initialise_graph_w_hidden_node = ndp_parameters['initialise_graph_w_hidden_node_flag']
        # Graph
        self.state_dim = ndp_parameters['state_dim']
        self.pruning_threshold = ndp_parameters['pruning_threshold']
        # Task
        self.n_inputs = task['n_inputs']
        self.n_outputs = task['n_outputs']
        # NDP Networks
        self.graph_cellular_automata = GraphCellularAutomata(self.state_dim, ndp_parameters['gca_hidden_size'])
        self.replication_model = ReplicationModel(self.state_dim, ndp_parameters['rm_hidden_size'])
        self.weight_prediction_model = WeightPredictionModel(self.state_dim, ndp_parameters['wp_hidden_size'])

    """
    Generate an initial graph.
    There are two options available, either connect all input nodes to the otuput nodes,
    or have a hidden node connected to every other node.
    """
    def generate_initial_seed_graph(self):
        # Create graph
        graph = Graph(self.weighted_graph_flag)

        # Add input nodes
        for _ in range(self.n_inputs):
            node = Node(state_dim=self.state_dim, node_type='input')
            graph.add_node(node)

        # Add output nodes
        for _ in range(self.n_outputs):
            node = Node(state_dim=self.state_dim, node_type='output')
            graph.add_node(node)
            if not self.initialise_graph_w_hidden_node:
                for i in range(self.n_inputs):
                    graph.add_edge(i, graph.node_id_count - 1)

        # Add hidden node (if option was selected)
        if self.initialise_graph_w_hidden_node:
            node = Node(state_dim=self.state_dim, node_type='hidden')
            graph.add_node(node)
            for i in range(self.n_inputs):
                    graph.add_edge(i, graph.node_id_count - 1)
            for i in range(self.n_outputs):
                    graph.add_edge(graph.node_id_count - 1, self.n_inputs + i)

        return graph

    """
    This perfroms the graph convolution. 
    The update is done by a Graph Cellular Automata.
    FIX: I am not sure what is the size of the hidden layer tbh.
    """
    def graph_convolution(self, graph:Graph, steps:int) -> Graph:
        for _ in range(steps):
            updated_nodes = list(graph.nodes)
            
            for i, node in enumerate(graph.nodes):
                _, neighbors_states = graph.get_neighbors_states(node.node_id)
                node_state = torch.tensor(node.state)
                neighbors_states = torch.tensor(neighbors_states)
                updated_nodes[i].state = self.graph_cellular_automata(node_state, neighbors_states).numpy()
            
            graph.nodes = list(updated_nodes)
        
        return graph


    def grow_graph(self, graph:Graph) -> Graph:
        # Replication model R determines nodes in growing state
        # New nodes are added to each of the growing nodes and their immediate neighbors
        # New nodes' embeddings are defined as the mean embeddings of their parent nodes

        new_nodes = [] 
        new_edges = {}
        for node in graph.nodes:
            if node.node_type == 'hidden':
                # Use the Replication model to decide if a node should be replicated
                state = torch.tensor(node.state, dtype=torch.float32)
                replicate_node = self.replication_model(state)
                # In case it does:
                if replicate_node:
                    # Create a new node
                    new_node_id = graph.node_id_count + len(new_nodes)
                    new_node = Node(node_id=new_node_id, state_dim=self.state_dim, node_type='hidden')
                    # Compute the mean of the neighbors
                    neighbor_ids, neighbors_states = graph.get_neighbors_states(node.node_id)
                    neighbors_states = np.vstack([neighbors_states, node.state])
                    mean_state = np.mean(neighbors_states, axis = 0)
                    new_node.state = np.expand_dims(mean_state, axis=0)
                    new_nodes.append(new_node)
                    # Add the edges to the new node
                    for id in neighbor_ids:
                        new_edges[(id, new_node_id)] = 1.0

        # Add the new nodes and edges to the graph
        graph.nodes.extend(new_nodes)
        graph.edges |= new_edges
        # Update the node id count
        graph.node_id_count += len(new_nodes)
        return graph
    
    def predict_weights(self, graph:Graph) -> Graph:
        # Weight update model W updates connectivity for each pair of nodes based on their concatenated embeddings

        # 1st version: only updates existing edges
        # for input_id, output_id in graph.edges.keys():
        #     input_node, output_node = graph.get_multiple_nodes([input_id, output_id])
        #     input_node_state = torch.tensor(input_node.state, dtype=torch.float32)
        #     output_node_state = torch.tensor(output_node.state, dtype=torch.float32)
        #     new_weight = self.weight_prediction_model(input_node_state, output_node_state).item()
        #     graph.edges[(input_id, output_id)] = new_weight
        # return graph

        # 2nd veresion: update values for all pair of nodes in the graph
        for input_node in graph.nodes:
            for output_node in graph.nodes:
                if not graph.is_this_edge_valid(input_node, output_node):
                    continue
                input_node_state = torch.tensor(input_node.state, dtype=torch.float32)
                output_node_state = torch.tensor(output_node.state, dtype=torch.float32)
                new_weight = self.weight_prediction_model(input_node_state, output_node_state).item()
                graph.edges[(input_node.node_id, output_node.node_id)] = new_weight
        return graph

    def prune(self, graph:Graph) -> Graph:
        # Edges with weights below pruning threshold P are removed
        
        # Find edges to remove
        edges_to_remove = []
        for edge, weight in graph.edges.items():
            if abs(weight) < self.pruning_threshold:
                edges_to_remove.append(edge)

        # Remove edges
        for edge in edges_to_remove:
            weight = graph.edges[edge]
            del graph.edges[edge]

            # Check that removed edge does not disconnect the outputs from the inputs
            if not graph.is_there_a_path_to_the_output():
                graph.edges[edge] = weight

        return graph

    def run_a_developmental_cycle(self, graph:Graph) -> Graph:
        # Compute network diameter D
        start_time = time.time()
        # print('A')
        diameter = graph.get_diameter()
        # print(f'Time = {time.time() - start_time}')
        # start_time = time.time()
        # print(diameter)

        # Propagate nodes states En via graph convolution D steps
        # print('B')
        graph = self.graph_convolution(graph, diameter)
        # print(f'Time = {time.time() - start_time}')
        # start_time = time.time()
        # graph.summary()

        # Replication model R determines nodes in growing state
        # New nodes are added to each of the growing nodes and their immediate neighbors
        # New nodes' embeddings are defined as the mean embeddings of their parent nodes
        # print('C')
        graph = self.grow_graph(graph)
        print(f'Time = {time.time() - start_time}')
        start_time = time.time()
        # graph.summary()

        # If weighted network then:
        print('D')
        if self.weighted_graph_flag:
            # Weight update model W updates connectivity for each pair of nodes based on their concatenated embeddings
            graph = self.predict_weights(graph)
        print(f'Time = {time.time() - start_time}')
        start_time = time.time()
        # graph.summary()

        # If pruning then
        if self.pruning_flag:
            # Edges with weights below pruning threshold P are removed
            graph = self.prune(graph)
        return graph

    def develope(self, n_cycles:int) -> Graph:
        with torch.no_grad():
            graph = self.generate_initial_seed_graph()
            print('Initial graph')
            graph.summary()
            for i in range(n_cycles):
                graph = self.run_a_developmental_cycle(graph)
                # print(f'Graph at cycle {i}')
                # graph.summary()
            return graph


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Main function (mainly for testing)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':

    import time
    n_cycles = 5
    ndp = NeuralDevelopmentalProgram()
    graph = ndp.develope(n_cycles)
    graph.summary()