'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
import numpy as np
from collections import deque

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
    REVIEW THIS FUNCTION LATER
    '''
    def update(self, neighbors:list[Node]):
        # How do I update the states????
        neighbors_states = [node.state for node in neighbors]
        neighbors_states.append(self.state)
        self.state = np.mean(neighbors_states, axis = 1)
        # I am really not sure that this is right!!!!



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
        self.node_id_count = 0

        self.nodes_count = {
            'input' : 0,
            'output' : 0,
            'hidden' : 0
        }        
        
        # This is IMPORTANT!!!! 
        # So I chose a list instead of a dict because node.id pairs the indices on the self.nodes parameter. 
        # If at any point, for some reason I start removing nodes, this will not make sense anymore. 
        self.adjacency = [] 

    def add_node(self, node:Node):
        node.node_id = self.node_id_count
        self.nodes.append(node)
        self.node_id_count += 1
        self.nodes_count[node.node_type] += 1

    def add_edge(self, input_node:int, output_node:int, weight:float = 1.0):
        self.edges[(input_node, output_node)] = weight if self.weighted_graph_flag else None

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

    def get_neighbors_states(self, node_id:int, update_adj:bool = False) -> list[int]:
        if update_adj:
            self.update_adjacency()
        # neighbors_ids = self.get_neighbors(node_id)
        neighbors_ids = self.adjacency[node_id]
        return neighbors_ids, self.get_nodes_states_by_id(neighbors_ids)

    def get_diameter(self, update_adj = True):
        if update_adj:
            self.update_adjacency()
        longest_path = [self._get_longest_path(node) for node in self.nodes]
        return max(longest_path)

    def _get_longest_path(self, node:Node):
        distances = self._bfs_distances(node)
        return max(distances.values())

    def get_diameter_in_parallel(self):
        from concurrent.futures import ProcessPoolExecutor
        CORES = 4
        with ProcessPoolExecutor(max_workers=CORES) as executor:
            longest_path = list(executor.map(self._get_longest_path_in_parallel, range(len(self.nodes))))
        return max(longest_path)

    def _get_longest_path_in_parallel(self, idx):
        distances = self._bfs_distances(self.nodes[idx])
        return max(distances.values())

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
    
    def summary(self):
        self.nodes = sorted(self.nodes, key = lambda x: x.node_id)
        print('-----------------')
        print('Nodes')
        print('-----------------')
        for node in self.nodes:
            print(f'id = {node.node_id}, type = {node.node_type}, state = {node.state}')
        print('\n-----------------')
        print('Edges')
        print('-----------------')
        for (input_node, output_node), weight in self.edges.items():
            print(f'input = {input_node}, output = {output_node}, weight = {weight}')
        print('\n')


    


