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

    # ---------------------------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------------------------

    def __init__(self, state_dim:int=None, weighted_graph_flag:bool=False, propagation_mode:str='undirected'):
        self.state_dim = state_dim
        self.weighted_graph_flag = weighted_graph_flag
        self._graph = nx.DiGraph()
        self.nodes_states = None
        self.propagation_mode = propagation_mode
        if propagation_mode not in ['undirected', 'directed']:
            raise ValueError("Propagation mode must be either 'undirected' or 'directed'")

    # ---------------------------------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------------------------------

    # Returns all nodes
    def nodes(self) -> list:
        return self._graph.nodes()

    # Returns the number of nodes
    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

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

    # ---------------------------------------------------------------------------------------
    # Edges
    # ---------------------------------------------------------------------------------------
    
    # Returns all edges
    def edges(self) -> list:
        return self._graph.edges()

    # Returns the number of nodes
    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    # Check if an edge exists    
    def has_edge(self, source:int, target:int) -> bool:
        return self._graph.has_edge(source, target)

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

    # ---------------------------------------------------------------------------------------
    # Adjacency 
    # ---------------------------------------------------------------------------------------

    # Returns the adjacency matrix
    def get_adjacency_matrix(self) -> np.array:
        return nx.to_numpy_array(self._graph, weight=None, dtype=bool)

    def get_weight_matrix(self) -> np.array:
        return nx.to_numpy_array(self._graph)

    # Updated the weight matrix 
    def update_weight_matrix(self, weight_matrix):
        # self._graph = nx.DiGraph(weight_matrix)
        for source, target in self._graph.edges():
            self._graph[source][target]["weight"] = float(weight_matrix[source, target])


    # ---------------------------------------------------------------------------------------
    # Neighbors
    # ---------------------------------------------------------------------------------------

    # Returns the neighbors of node_id 
    def get_neighbors(self, node_id) -> set:
        return set(nx.all_neighbors(self._graph, node_id))
    
    def successors(self, node_id) -> set:
        return set(self._graph.successors(node_id))

    # ---------------------------------------------------------------------------------------
    # Graph Stats
    # ---------------------------------------------------------------------------------------

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

    # Return the longest finite directed shortest-path distance.
    def get_maximum_directed_distance(self) -> int:
        distances = (
            distance
            for _, target_distances in nx.all_pairs_shortest_path_length(self._graph)
            for distance in target_distances.values()
        )
        return max(distances, default=0)

    def get_propagation_distance(self) -> int:
        if self.propagation_mode == 'directed':
            return self.get_maximum_directed_distance()
        return self.get_largest_subgraph_diameter()    

    def get_unreachable_outputs(self, n_inputs: int, n_outputs: int) -> list:
        input_nodes = range(n_inputs)
        output_nodes = range(self.number_of_nodes() - n_outputs, self.number_of_nodes())
        unreachable_outputs = [output_node for output_node in output_nodes if not any(nx.has_path(self._graph, input_node, output_node) for input_node in input_nodes)]
        return unreachable_outputs

    def are_all_outputs_reachable(self, n_inputs:int, n_outputs:int) -> bool:
        return len(self.get_unreachable_outputs(n_inputs, n_outputs)) == 0



    # ---------------------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------------------
    
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




        
    
