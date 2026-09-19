'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Libraries
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

from __future__ import annotations
import numpy as np
import networkx as nx

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp

'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Graph (with nx.Graph() and JAX)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        'nodes_states',
        'node_mask',
        'adjacency',
        'weights'
    ],
    meta_fields=[
        'max_nodes',
        'state_dim',
        'weighted_graph_flag',
    ],
)
@dataclass(frozen=True)
class GraphJax():

    # JAX arrays
    nodes_states:jax.Array
    node_mask:jax.Array
    adjacency:jax.Array
    weights:jax.Array

    # Static information
    max_nodes:int
    state_dim:int
    weighted_graph_flag:bool = False

    # ---------------------------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------------------------

    @classmethod
    def create(cls, max_nodes:int, state_dim:int, weighted_graph_flag:bool = False):
        nodes_states = jnp.zeros((max_nodes, state_dim), dtype=jnp.float32)
        node_mask = jnp.zeros(max_nodes, dtype=jnp.bool_)
        adjacency = jnp.zeros((max_nodes, max_nodes), dtype=jnp.bool_)
        weights = jnp.zeros((max_nodes, max_nodes), dtype=jnp.float32)
        return cls(
            nodes_states = nodes_states,
            node_mask = node_mask,
            adjacency = adjacency,
            weights = weights,
            max_nodes = max_nodes,
            state_dim = state_dim, 
            weighted_graph_flag = weighted_graph_flag 
        )

    # ---------------------------------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------------------------------

    # Returns all nodes IDs (inactive nodes are represented with a -1)
    def nodes(self) -> jax.Array:
        return jnp.where(self.node_mask, size=self.max_nodes, fill_value=-1)[0]

    # Returns the number of nodes
    def number_of_nodes(self) -> int:
        return jnp.sum(self.node_mask)

    # Adds a single node
    def add_node(self, new_node_state:np.array) -> tuple:
        new_node_id = self.number_of_nodes()
        nodes_states = self.nodes_states.at[new_node_id].set(new_node_state)
        node_mask = self.node_mask.at[new_node_id].set(True)
        graph = replace(self, nodes_states=nodes_states, node_mask=node_mask)
        return graph, new_node_id
    
    # Adds multiple nodes
    def add_nodes_from(self, new_nodes_states:np.array) -> GraphJax:
        new_nodes_states = jnp.asarray(new_nodes_states)
        n_new_nodes = new_nodes_states.shape[0]
        first_node = self.number_of_nodes()
        nodes_states = jax.lax.dynamic_update_slice(self.nodes_states, new_nodes_states, (first_node, 0)) 
        new_mask = jnp.ones(n_new_nodes, dtype=jnp.bool_)
        node_mask = jax.lax.dynamic_update_slice(self.node_mask, new_mask, (first_node,)) 
        return replace(self, nodes_states=nodes_states, node_mask=node_mask)


    # ---------------------------------------------------------------------------------------
    # Edges
    # ---------------------------------------------------------------------------------------
    
    # Returns all edges
    def edges(self) -> jax.Array:
        source, target = jnp.nonzero(
            self.adjacency
            & self.node_mask[:, None]
            & self.node_mask[None, :]
        )
        return jnp.stack([source, target], axis=1)

    # Returns the number of nodes
    def number_of_edges(self) -> int:
        active_edges = self.adjacency & self.node_mask[:, None] & self.node_mask[None, :]
        return jnp.sum(active_edges)

    # Check if an edge exists    
    def has_edge(self, source:int, target:int) -> bool:
        return self.adjacency[source, target]

    # Changes edge state on adjacency matrix and weight value
    def _change_edge(self, input_id:int, output_id:int, edge_state:bool, weight:float):
        adjacency = self.adjacency.at[input_id, output_id].set(edge_state)
        weights = self.weights.at[input_id, output_id].set(weight)
        return replace(self, adjacency=adjacency, weights=weights)

    # Changes multiple edges states on adjacency matrix and weights value
    def _change_multiple_edges(self, edges:list, edges_state:bool, weight:float):
        edges = jnp.asarray(edges)
        input_ids = edges[:, 0]
        output_ids = edges[:, 0]
        adjacency = self.adjacency.at[input_ids, output_ids].set(edges_state)
        weights = self.weights.at[input_ids, output_ids].set(weight)
        return replace(self, adjacency=adjacency, weights=weights)

    # Adds a single edge
    def add_edge(self, input_id:int, output_id:int, weight:float = 1.0):
        return self._change_edge(input_id, output_id, edge_state=True, weight=weight)

    # Adds multiple edges
    def add_edges_from(self, new_edges:list, weight:float = 1.0):
        return self._change_multiple_edges(new_edges, edges_state=True, weight=weight)

    # Removes an edge 
    def remove_edge(self, input_id, output_id):
        return self._change_edge(input_id, output_id, edge_state=False, weight=0.0)

    # Removes multiple edges
    def remove_edges_from(self, edges:list):
        return self._change_multiple_edges(edges, edges_state=False, weight=0.0)

    # Adds multiple edges from defined source nodes to target nodes.
    # Density 1.0 is fully connected, lower to 1.0 and higher to 0.0 is sparse
    def add_sparse_edges(self, source_nodes:np.array, target_nodes:np.array, density:float, key, allow_self_loops:bool=False):
        source_nodes = jnp.asarray(source_nodes)
        target_nodes = jnp.asarray(target_nodes)

        mask = jax.random.uniform(key, shape=(source_nodes.shape[0], target_nodes.shape[0])) < density

        if not allow_self_loops:
            mask &= source_nodes[:,None] != target_nodes[None, :]

        current = self.adjacency[source_nodes[:, None], target_nodes[None, :]]
        current_weights = self.weights[source_nodes[:, None], target_nodes[None, :]]

        updated = current | mask
        updated_weights = jnp.where(mask & ~current, 1.0, current_weights)

        adjacency = self.adjacency.at[source_nodes[:, None], target_nodes[None, :]].set(updated)
        weights = self.weights.at[source_nodes[:, None], target_nodes[None, :]].set(updated_weights)

        return replace(self, adjacency=adjacency, weights=weights)

    # ---------------------------------------------------------------------------------------
    # Adjacency
    # ---------------------------------------------------------------------------------------

    # Returns the adjacency matrix
    def get_adjacency_matrix(self) -> np.array:
        if self.weighted_graph_flag:
            return self.weights
        return self.adjacency.astype(jnp.float32)

    # Updated the adjacency matrix 
    def update_adjacency_matrix(self, adjacency_matrix):
        adjacency_matrix = jnp.asarray(adjacency_matrix)
        adjacency = adjacency_matrix.astype(jnp.bool_)
        weights = adjacency_matrix.astype(jnp.float32)
        return replace(self, adjacency=adjacency, weights=weights)

    # ---------------------------------------------------------------------------------------
    # Neighbors
    # ---------------------------------------------------------------------------------------

    def neighbor_mask(self, node_id):
        outgoing = self.adjacency[node_id, :]
        incoming = self.adjacency[:, node_id]
        return ((outgoing | incoming) & self.node_mask)

    # Returns the neighbors of node_id 
    def get_neighbors(self, node_id):
        return jnp.where(self.neighbor_mask(node_id), size=self.max_nodes, fill_value=-1,)[0]

    def successor_mask(self, node_id):
        return (self.adjacency[node_id] & self.node_mask)

    def successors(self, node_id):
        return jnp.where(self.successor_mask(node_id), size=self.max_nodes, fill_value=-1,)[0]

    def get_neighbor_matrix(self, include_self_node:bool = False):
        neighbor_matrix =  self.adjacency | self.adjacency.T

        if not include_self_node:
            return neighbor_matrix

        neighbor_matrix = neighbor_matrix | jnp.eye(self.max_nodes, dtype=jnp.bool_)
        neighbor_matrix &= (self.node_mask[:, None] & self.node_mask[None, :])
        return neighbor_matrix

    # ---------------------------------------------------------------------------------------
    # NetworkX Functions
    # ---------------------------------------------------------------------------------------

    def to_networkx(self):
        n_nodes = self.number_of_nodes()
        adjacency = np.asarray(self.get_adjacency_matrix())[:n_nodes, :n_nodes]
        return nx.from_numpy_array(adjacency, create_using=nx.DiGraph)

    # Returns the diameter
    def get_diameter(self) -> int:
       return nx.diameter(self.to_networkx().to_undirected())

    # Returns the diameter of the largest subgraph (in case not all nodes are connected) 
    def get_largest_subgraph_diameter2(self) -> int:
        graph = self.to_networkx().to_undirected()

        if nx.is_connected(graph):
            return nx.diameter(graph)
        
        largest_component_nodes = max(nx.connected_components(graph), key=len)
        largest_component = graph.subgraph(largest_component_nodes)
        return nx.diameter(largest_component)

    def get_largest_subgraph_diameter(self) -> jax.Array:

        # Convert directed graph to undirected
        adjacency = self.adjacency | self.adjacency.T

        # Only consider active nodes
        active_pair_mask = (
            self.node_mask[:, None]
            & self.node_mask[None, :]
        )

        adjacency = adjacency & active_pair_mask

        # -----------------------------------------
        # Initialise shortest-path matrix
        # -----------------------------------------

        inf = jnp.inf

        distances = jnp.where(
            adjacency,
            1.0,
            inf
        )

        # Distance from an active node to itself = 0
        indices = jnp.arange(self.max_nodes)

        distances = distances.at[
            indices,
            indices
        ].set(
            jnp.where(
                self.node_mask,
                0.0,
                inf
            )
        )

        # -----------------------------------------
        # Floyd-Warshall
        # -----------------------------------------

        def floyd_step(k, distances):

            through_k = (
                distances[:, k, None]
                + distances[None, k, :]
            )

            return jnp.minimum(
                distances,
                through_k
            )

        distances = jax.lax.fori_loop(
            0,
            self.max_nodes,
            floyd_step,
            distances
        )

        # -----------------------------------------
        # Find largest connected component
        # -----------------------------------------

        reachable = jnp.isfinite(
            distances
        )

        component_sizes = jnp.sum(
            reachable,
            axis=1
        )

        # Inactive nodes shouldn't be candidates
        component_sizes = jnp.where(
            self.node_mask,
            component_sizes,
            -1
        )

        # Any node belonging to the largest component
        component_root = jnp.argmax(
            component_sizes
        )

        largest_component_mask = reachable[
            component_root
        ]

        # -----------------------------------------
        # Diameter of largest component
        # -----------------------------------------

        pair_mask = (
            largest_component_mask[:, None]
            & largest_component_mask[None, :]
        )

        component_distances = jnp.where(
            pair_mask,
            distances,
            0.0
        )

        diameter = jnp.max(
            component_distances
        )

        return diameter.astype(jnp.int32)
    

    # ---------------------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------------------

    # Prints a summary of the graph 
    def summary(self, full:bool=False):
        n_nodes = int(np.asarray(self.number_of_nodes()))
        n_edges = int(np.asarray(self.number_of_edges()))

        states = np.asarray(
            self.nodes_states[:n_nodes]
        )

        adjacency = np.asarray(
            self.get_adjacency_matrix()
        )[:n_nodes, :n_nodes]

        print("------------------------------------")
        print("Graph")
        print("------------------------------------")
        print(f"Total number of nodes = {n_nodes}")
        print(f"Total number of edges = {n_edges}")

        if n_nodes > 0:
            print(f"Min state = {states.min(axis=0)}")
            print(f"Max state = {states.max(axis=0)}")
            print(f"Mean state = {states.mean(axis=0)}")
            print(f"Std state = {states.std(axis=0)}")

        if full:

            print("------------------------------------")
            print("Nodes")
            print("------------------------------------")

            for i, node_state in enumerate(states):
                print(f"{i} = {node_state}")

            print("\n------------------------------------")
            print("Edges")
            print("------------------------------------")

            sources, targets = np.nonzero(adjacency)

            for source, target in zip(sources, targets):
                print(
                    f"({source}, {target}) = "
                    f"{adjacency[source, target]}"
                )

        print()