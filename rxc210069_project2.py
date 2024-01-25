import itertools
import matplotlib.pyplot as plt

class NetworkGraph:
    def __init__(self, size):
        # Initialize the network graph with the specified number of nodes.
        self.size = size  # Size refers to the number of nodes in the graph.
        # Create a complete graph, where each node is connected to all other nodes (except itself).
        self.graph = {i: {j for j in range(size) if j != i} for i in range(size)}
        # Identify all unique triangles within the graph. A triangle is a set of three interconnected nodes.
        self.triangles = self._find_triangles()

    def _find_triangles(self):
        # This private method identifies all unique triangles in the graph.
        # It uses itertools.combinations to find all possible pairs of neighbors for each node.
        # Then checks if those pairs are interconnected, indicating a triangle.
        return {tuple(sorted((node, *pair)))
                for node in self.graph
                for pair in itertools.combinations(self.graph[node], 2)
                if pair[0] in self.graph[pair[1]]}

    def is_operational(self, simulated_graph):
        # This method checks if the network is operational.
        # A network is operational if each node has at least one active connection to another node.
        return all(neighbors for neighbors in simulated_graph.values())

    def simulate_network(self, state):
        # Simulate the network based on the operational state of triangles.
        # 'state' is a sequence indicating which triangles are operational (True) or not (False).
        simulated_graph = {node: set(neighbors) for node, neighbors in self.graph.items()}
        for triangle, operational in zip(self.triangles, state):
            if not operational:
                # If a triangle is not operational, remove its connections from the graph.
                # This simulates the effect of a triangle (set of connections) being down.
                simulated_graph = {node: neighbors - set(triangle) 
                                   if node in triangle else neighbors
                                   for node, neighbors in simulated_graph.items()}
        return simulated_graph

    def calculate_reliability(self, p):
        # Calculate the overall reliability of the network given the probability 'p'.
        # Reliability is computed over all possible states of the triangles.
        # Uses the principle of Bernoulli trials for each triangle's operational state.
        return sum((p ** sum(state)) * ((1 - p) ** (len(state) - sum(state)))
                   for state in itertools.product([True, False], repeat=len(self.triangles))
                   if self.is_operational(self.simulate_network(state)))

def plot_reliability(p_values, reliabilities):
    # Function to plot the relationship between probability and network reliability.
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, reliabilities, 'o-', color='darkorange', label='Network Reliability')
    plt.xlabel('Probability (p)', fontsize=14)  # Label for the x-axis.
    plt.ylabel('Network Reliability', fontsize=14)  # Label for the y-axis.
    plt.title('Network Reliability vs Probability', fontsize=16)  # Title of the plot.
    plt.grid(True)  # Enable grid for better readability.
    plt.legend()  # Show legend.
    plt.xticks(p_values, rotation=45)  # Rotate x-axis labels for clarity.
    plt.tight_layout()  # Adjust layout for no overlap.
    plt.show()  # Display the plot.

# Main execution block
network_size = 5  # Define the size (number of nodes) for the network graph.
network = NetworkGraph(network_size)  # Instantiate the NetworkGraph class.
p_values = [i / 20 for i in range(1, 21)]  # Create a range of probability values from 0.05 to 1.0.
reliabilities = [network.calculate_reliability(p) for p in p_values]  # Calculate reliability for each probability value.

# Call the function to plot the network reliability against different probability values.
plot_reliability(p_values, reliabilities)
