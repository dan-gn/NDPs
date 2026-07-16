import numpy as np

from Graph.graph import Graphnx
from NDP.ndp_nx import NeuralDevelopmentalProgram

class HebbianNeuralDevelopmentalProgram(NeuralDevelopmentalProgram):


    def __init__(self, config:dict = None):
        super().__init__(config)

    def _set_config(self, config):
        super()._set_config(config)
        self.n_nodes = config['n_nodes']
        self.initial_graph_density = config['initial_graph_density']

    def _check_valid_config(self):
        super()._check_valid_config()
        if self.graph_n_inputs + self.graph_n_outputs > self.n_nodes:
            raise ValueError('The number of inputs plus outputs exceed the total number of nodes.')

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
        if self.initial_node_state_mode == 'coevolve':
            coevolved_array = self.shared_initial_node_state.copy()
            network_seed = int(np.clip(coevolved_array[0, 0] + 1, 0, 2) * 100)
            nodes_states = coevolved_array[0, 1:]
            nodes_states = nodes_states.reshape(self.n_nodes, self.state_dim)
            graph.add_nodes_from(nodes_states)

        elif self.initial_node_state_mode == 'random_shared':
            raise NotImplementedError("Come on, Daniel! You haven't implemented this yet! Stop procrastinating and just do it!")

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
        rng = np.random.default_rng(network_seed)
        density = self.initial_graph_density

        # Add edges from inputs to outputs nodes (Fully connected) 
        graph.add_sparse_edges(input_nodes, output_nodes, density=1.0, rng=rng)

        # Add edges from inputs to hidden nodes (sparse)
        graph.add_sparse_edges(input_nodes, hidden_nodes, density=density, rng=rng)

        # Add edges from hidden to hidden nodes (sparse)
        graph.add_sparse_edges(hidden_nodes, hidden_nodes, density=density, rng=rng)

        # Add edges from hidden to output nodes (sparse)
        graph.add_sparse_edges(hidden_nodes, output_nodes, density=density, rng=rng)

        return graph

    def structural_synapsis(self, graph:Graphnx) -> Graphnx:
        """
        Here we are gonna modify the ANN structure. We should be able to:
        1. Add new edges - (Randomly sample possible new edges)
        2. Prune edges - (Check all edges)
        3. Split edges and adding new neurons - (Should I do this?)

        """
        return graph
    
    # def _run_a_developmental_cycle(self, graph:Graphnx) -> Graphnx:
    #     """
    #     This method runs one developmental cycle.
    #     Steps are as follow:
    #     1. Compute graph diameter
    #     2. Graph convolution
    #     3. Structural synapsis
    #     """
    #     # Compute network diameter D
    #     diameter = graph.get_diameter()

    #     # Propagate nodes states En via graph convolution D steps
    #     steps = diameter + self.network_extra_thinking
    #     graph = self.graph_convolution(graph, steps)

    #     # Structural Synapsis (Add and remove edges)
    #     graph = self.structural_synapsis(graph)

    #     return graph
