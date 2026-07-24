'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

from __future__ import annotations
import numpy as np
import random
from collections import deque, defaultdict

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Single node
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class Node:

    def __init__(self, node_id = None, state_dim = 10, node_type = 'hidden'):
        self.node_id = node_id
        self.state_dim = state_dim   
        self.state = np.zeros((1, state_dim), np.float32)  
        self.node_type = node_type
        self.neighbors = []


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Graph
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class Graph:

    def __init__(self, weighted_graph_flag = False):
        self.nodes = []
        self.edges = {}

        self.weighted_graph_flag = weighted_graph_flag

        self.nodes_count = {
            'input' : 0,
            'output' : 0,
            'hidden' : 0
        }        
        
        self.adjacency = defaultdict(list)

    def add_node(self, node:Node):
        node.node_id = len(self.nodes)
        self.nodes.append(node)
        self.nodes_count[node.node_type] += 1

    def add_edge(self, input_node:int, output_node:int, weight:float = 1.0):
        self.edges[(input_node, output_node)] = weight if self.weighted_graph_flag else None
        self.adjacency[input_node].append(output_node)
        self.adjacency[output_node].append(input_node)

    def delete_edge(self, input_node:int, output_node:int):
        if (input_node, output_node) in self.edges:
            del self.edges[(input_node, output_node)]
            self.adjacency[input_node].remove(output_node)
            self.adjacency[output_node].remove(input_node)

    def get_node(self, node_id:int) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_multiple_nodes(self, list_of_node_ids:list[int]) -> list[Node]:
        return [node for node in self.nodes if node.node_id in list_of_node_ids]
    
    def update_adjacency(self):
        self.adjacency = [list(self.get_neighbors(node.node_id)) for node in self.nodes]

    def get_neighbors(self, node_id:int) -> list[int]:
        neighbors = []
        for (input_node, output_node), _ in self.edges.items():
            if input_node == node_id:
                neighbors.append(output_node)
            elif output_node == node_id:
                neighbors.append(input_node)
        return neighbors

    def get_nodes_states_by_id(self, list_of_node_ids:list[int]) -> list:
        states = [node.state for node in self.nodes if node.node_id in list_of_node_ids]
        if len(states) > 1:
            return np.concatenate(states, axis=0)
        elif len(states) == 1:
            return np.array(states).squeeze(axis=0)
        else:
            return np.array(states)

    def get_neighbors_states(self, node_id:int) -> list[int]:
        # neighbors_ids = self.get_neighbors(node_id)
        neighbors_ids = self.adjacency[node_id]
        return neighbors_ids, self.get_nodes_states_by_id(neighbors_ids)

    def get_diameter(self):
        if len(self.nodes) <= 10:
            longest_path = max([self._get_longest_path(node) for node in self.nodes])
        else:
            longest_path = self.get_diameter_multi_sweep()
        return longest_path

    def _get_longest_path(self, node:Node):
        distances = self._bfs_distances(node)
        return max(distances.values())

    def get_diameter_in_parallel(self):
        from concurrent.futures import ProcessPoolExecutor
        cores = 4
        with ProcessPoolExecutor(max_workers=cores) as executor:
            longest_path = list(executor.map(self._get_longest_path_in_parallel, range(len(self.nodes))))
        return max(longest_path)

    def _get_longest_path_in_parallel(self, idx):
        distances = self._bfs_distances(self.nodes[idx])
        return max(distances.values())
    
    def get_diameter_multi_sweep(self, k=10):
        sample_nodes = random.sample(self.nodes, k=min(k, len(self.nodes)))
        diameter_estimate = 0

        for node in sample_nodes:
            farthest_node, _ = self._bfs_farthest_node(node.node_id)
            _, distance = self._bfs_farthest_node(farthest_node)
            diameter_estimate = max(diameter_estimate, distance)

        return diameter_estimate
    
    def _bfs_farthest_node(self, start_id):
        distances = {start_id: 0}
        queue = deque([start_id])
        farthest_node = start_id
        max_distance = 0

        while queue:
            current_id = queue.popleft()
            current_distance = distances[current_id]

            if current_distance > max_distance:
                max_distance = current_distance
                farthest_node = current_id

            for neighbor_id in self.adjacency[current_id]:
                if neighbor_id not in distances:
                    distances[neighbor_id] = current_distance + 1
                    queue.append(neighbor_id)

        return farthest_node, max_distance

    def _bfs_distances(self, start_node:Node):
        start_node_id = start_node.node_id

        distances = {start_node_id : 0} # This will store the distance from the start node to this node
        queue = deque([start_node_id]) # This is the queue of nodes to explore

        while queue:
            current_id = queue.popleft()    # Get a node to explore
            current_distance = distances[current_id]    # Current distance to this node

            # neighbors = self.get_neighbors(current_id)
            neighbors = self.adjacency[current_id]

            for neighbor_id in neighbors:
                if neighbor_id not in distances:    # If this node has not being explored
                    distances[neighbor_id] = current_distance + 1   # It is one node away from the current node
                    queue.append(neighbor_id)   # And we should explore it in the future
        return distances
    
    def _get_directed_neighbors(self, node_id):
        neighbors = []
        for input_id, output_id in self.edges.keys():
            if input_id == node_id:
                neighbors.append(output_id)
        return neighbors

    def is_there_a_path_to_the_output(self):
        # Get all input and output nodes
        # input_nodes = [node.node_id for node in self.nodes if node.node_type == 'input']
        # output_nodes = [node.node_id for node in self.nodes if node.node_type == 'output']
        input_nodes = [i for i in range(self.nodes_count['input'])]
        output_nodes = [i + self.nodes_count['input'] for i in range(self.nodes_count['output'])]
        # Visited set, out of loop cause I just want to know that all outputs can be reached by at least one input
        visited = set()
        # BFS for all input nodes
        for input_node in input_nodes:
            queue = deque([input_node])
            while queue:
                current_id = queue.popleft()
                if current_id in visited:
                    continue
                else:
                    visited.add(current_id)
                # Get directed neighbors (this applies for when the network is weighted)
                # FIX: Add version for when the neighbors are not weighted
                neighbors = self._get_directed_neighbors(current_id)
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
                    if neighbor_id in output_nodes:
                        output_nodes.remove(neighbor_id)
                        # Check if all output nodes had been reached
                        if not output_nodes:
                            # Finish the function if it has already happend
                            return True
        return False

    def is_this_edge_valid(self, input_node, output_node):
        if input_node == output_node:
            return False

        if input_node.node_type == 'output':
            return False
        
        if output_node.node_type == 'input':
            return False
        
        return True
     
    def copy(self):
        pass
    
    def summary(self, full:bool=False):
        self.nodes = sorted(self.nodes, key = lambda x: x.node_id)
        print('------------------------------------')
        print('Graph')
        print('------------------------------------')
        print(f'Total number of nodes = {len(self.nodes)}')
        print(f"Number of input nodes = {self.nodes_count['input']}")
        print(f"Number of output nodes = {self.nodes_count['output']}")
        print(f"Number of hidden nodes = {self.nodes_count['hidden']}")
        print(f'Number of edges = {len(self.edges)}')
        if full:
            print('------------------------------------')
            print('Nodes')
            print('------------------------------------')
            for node in self.nodes:
                print(f'id = {node.node_id}, type = {node.node_type}, state = {node.state}')
            print('\n------------------------------------')
            print('Edges')
            print('------------------------------------')
            for (input_node, output_node), weight in self.edges.items():
                print(f'input = {input_node}, output = {output_node}, weight = {weight}')
        print('\n')


    


'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Graph (with nx.Graph())
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
import networkx as nx

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
    def add_sparse_edges(self, source_nodes:list, target_nodes:list, density:float, rng:np.random.Generator, allow_self_loops:bool=False):
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




        
    

