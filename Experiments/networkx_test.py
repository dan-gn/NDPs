import networkx as nx

graph = nx.DiGraph()

# Add an edge with a normal, non-zero weight.
graph.add_edge("A", "B", weight=5)
print("After adding the edge:")
print("  Edge exists:", graph.has_edge("A", "B"))
print("  Edge data:  ", graph["A"]["B"])
print(nx.to_numpy_array(graph, weight=None, dtype=bool))

# Change the existing edge's weight to zero.
graph["A"]["B"]["weight"] = 0
print("\nAfter setting its weight to 0:")
print("  Edge exists:", graph.has_edge("A", "B"))
print("  Edge data:  ", graph["A"]["B"])
print("  Number of edges:", graph.number_of_edges())
print(nx.to_numpy_array(graph, weight=None, dtype=bool))

# A zero-weight edge still participates in weighted path calculations.
print(
    "  Weighted shortest-path length from A to B:",
    nx.shortest_path_length(graph, "A", "B", weight="weight"),
)

print("\nConclusion: weight=0 does not remove or disable the edge.")
print("Use graph.remove_edge('A', 'B') if you want to remove it.")
