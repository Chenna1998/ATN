import random
import matplotlib.pyplot as plt
import networkx as nx

# Constants
NUM_NODES = 25
NUM_ITERATIONS = 10

# --- GRAPH OPERATIONS ---

class GraphOperations:
    @staticmethod
    def generate(p):
        """Generate a random undirected graph with NUM_NODES nodes."""
        graph = {i: set() for i in range(NUM_NODES)}
        for i in range(NUM_NODES):
            for j in range(i+1, NUM_NODES):
                if random.random() < p:
                    graph[i].add(j)
                    graph[j].add(i)
        return graph

    @staticmethod
    def extract_k_core(graph, k):
        """Extract the k-core from the graph."""
        while True:
            nodes_to_prune = [node for node, neighbors in graph.items() if len(neighbors) < k]
            if not nodes_to_prune:
                break
            for node in nodes_to_prune:
                for neighbor in list(graph[node]):
                    if neighbor in graph:
                        graph[neighbor].discard(node)
                del graph[node]
        return graph

    @staticmethod
    def determine_core_value(graph):
        """Determine the core value of the graph."""
        k = 1
        while GraphOperations.extract_k_core(graph.copy(), k):
            k += 1
        return k - 1

# --- VISUALIZATION FUNCTIONS ---

def visualize(graph, p):
    """Visualize the graph using NetworkX."""
    k = GraphOperations.determine_core_value(graph)
    core_graph = GraphOperations.extract_k_core(graph, k)
    G = nx.Graph(core_graph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, with_labels=True, node_color='lightblue', node_size=1000, width=2.0, alpha=0.7)
    plt.title(f'Graph Visualization (p={p}, Core={k})')
    plt.show()

def plot_relation(probabilities, core_values):
    """Plot the relationship between edge probability and core value."""
    plt.plot(probabilities, core_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('Edge Probability')
    plt.ylabel('Average Core Value')
    plt.title('Relationship between Edge Probability and Core Value')
    plt.grid(True)
    plt.show()

# --- MAIN EXECUTION ---

def main():
    """Main execution function."""
    sample_prob = 0.5
    avg_core = sum(GraphOperations.determine_core_value(GraphOperations.generate(sample_prob)) for _ in range(NUM_ITERATIONS)) / NUM_ITERATIONS
    print(f"Sample: Average Core for p={sample_prob} is {avg_core}")
    
    for prob in [0.15, 0.5, 0.85]:
        visualize(GraphOperations.generate(prob), prob)
    
    probabilities = [i/100 for i in range(5, 100, 5)]
    core_values = [sum(GraphOperations.determine_core_value(GraphOperations.generate(p)) for _ in range(NUM_ITERATIONS)) / NUM_ITERATIONS for p in probabilities]
    plot_relation(probabilities, core_values)

if __name__ == "__main__":
    main()
