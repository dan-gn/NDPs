'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

from __future__ import annotations
import numpy as np
import networkx as nx

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Graph (with nx.Graph())
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class Graphnx():

    def __init__(self, state_dim:int=None, weighted_graph_flag:bool=False):
        self.state_dim = state_dim
        self.weighted_graph_flag = weighted_graph_flag
        self._graph = nx.DiGraph()
        self.nodes_states = None

    # Returns the diameter
    def get_diameter(self) -> int:
       return nx.diameter(self._graph.to_undirected())

    # Returns the diameter of the largest subgraph (in case not all nodes are connected) 
    def get_largest_subgraph_diameter(self) -> int:
        graph = self._graph.to_undirected()

        if nx.is_connected(graph):
            return nx.diameter(graph)
        
        largest_component_nodes = max(nx.connected_components(graph), key=len)
        largest_component = graph.subgraph(largest_component_nodes)
        return nx.diameter(largest_component)

    # Returns all nodes
    def nodes(self) -> list:
        return self._graph.nodes()

    # Returns the number of nodes
    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    # Returns all edges
    def edges(self) -> list:
        return self._graph.edges()

    # Returns the number of nodes
    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    # Check if an edge exists    
    def has_edge(self, source:int, target:int) -> bool:
        return self._graph.has_edge(source, target)

    # Returns the adjacency matrix
    def get_adjacency_matrix(self) -> np.array:
        return nx.to_numpy_array(self._graph)

    # Updated the adjacency matrix 
    def update_adjacency_matrix(self, adjacency_matrix):
        self._graph = nx.DiGraph(adjacency_matrix)

    # Returns the neighbors of node_id 
    def get_neighbors(self, node_id) -> set:
        return set(nx.all_neighbors(self._graph, node_id))
    
    def successors(self, node_id) -> set:
        return set(self._graph.successors(node_id))

    # Adds a single node
    def add_node(self, new_node_state:np.array) -> int:
        if self.nodes_states is None:
            node_id = 0
            self.nodes_states = new_node_state
        else:
            node_id = self._graph.number_of_nodes()
            self.nodes_states = np.vstack([self.nodes_states, new_node_state])
        self._graph.add_node(node_id)
        return node_id
    
    # Adds multiple nodes
    def add_nodes_from(self, node_states:np.array):
        if self.nodes_states is None:
            nodes_id = range(len(node_states))
            self.nodes_states = node_states.copy()
        else:
            last_id = self._graph.number_of_nodes()
            nodes_id = range(last_id, last_id + len(node_states))
            self.nodes_states = np.vstack([self.nodes_states, node_states])
        self._graph.add_nodes_from(nodes_id)

    # Adds a single edge
    def add_edge(self, input_node:int, output_node:int):
        self._graph.add_edge(input_node, output_node)

    # Adds multiple edges
    def add_edges_from(self, edges:list):
        self._graph.add_edges_from(edges)

    # Removes multiple edges
    def remove_edges_from(self, edges:list):
        self._graph.remove_edges_from(edges)

    # Adds multiple edges from defined source nodes to target nodes.
    # Density 1.0 is fully connected, lower to 1.0 and higher to 0.0 is sparse
    def add_sparse_edges(self, source_nodes:np.array, target_nodes:np.array, density:float, rng:np.random.Generator, allow_self_loops:bool=False):
        source_nodes = np.asarray(source_nodes)
        target_nodes = np.asarray(target_nodes)

        mask = rng.random((len(source_nodes), len(target_nodes))) < density

        if not allow_self_loops:
            mask &= source_nodes[:,None] != target_nodes[None, :]

        source_indices, target_indices = np.nonzero(mask)

        self.add_edges_from(zip(source_nodes[source_indices], target_nodes[target_indices]))

    # Removes an edge 
    def remove_edge(self, input_id, output_id):
        self._graph.remove_edge(input_id, output_id)

    # Prints a summary of the graph 
    def summary(self, full:bool=False):
        nodes = list(self.nodes())
        edges = list(self.edges())
        weights = self.get_adjacency_matrix()
        print('------------------------------------')
        print('Graph')
        print('------------------------------------')
        print(f'Total number of nodes = {len(nodes)}')
        print(f'Total number of edges = {len(edges)}')
        print(f'Min state = {self.nodes_states.min(axis=0)}')
        print(f'Max state = {self.nodes_states.max(axis=0)}')
        print(f'Mean state = {self.nodes_states.mean(axis=0)}')
        print(f'Std state = {self.nodes_states.std(axis=0)}')
        if full:
            print('------------------------------------')
            print('Nodes')
            print('------------------------------------')
            for i, node_state in enumerate(self.nodes_states):
                print(f'{i} = {node_state}')
            print('\n------------------------------------')
            print('Edges')
            print('------------------------------------')
            for i, (input_id, output_id) in enumerate(edges):
                print(f'({input_id, output_id}) = {weights[input_id, output_id]}')
        print('\n')




        
    
