from Graph.ndp_graph import Graphnx
from NDP.ndp_nx import NeuralDevelopmentalProgram

class HebbianNeuralDevelopmentalPrograms(NeuralDevelopmentalProgram):


    def __init__(self, config:dict = None):
        super().__init__(config)

    def graph_convolution(self, graph, steps):
        return super().graph_convolution(graph, steps)
    
    def structural_synapsis(self, graph:Graphnx) -> Graphnx:
        return graph
    
    def _run_a_developmental_cycle(self, graph:Graphnx) -> Graphnx:
        """
        This method runs one developmental cycle.
        Steps are as follow:
        1. Compute graph diameter
        2. Graph convolution
        3. Structural synapsis
        """
        # Compute network diameter D
        diameter = graph.get_diameter()

        # Propagate nodes states En via graph convolution D steps
        graph = self.graph_convolution(graph, diameter)

        # Structural Synapsis (Add and remove edges)
        graph = self.structural_synapsis(graph)

        return graph
