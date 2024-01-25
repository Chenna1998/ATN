import random
import matplotlib.pyplot as plt
import networkx as nx

# Constants
NUM_NODES = 25
NUM_ITERATIONS = 10

# --- UTILITY FUNCTIONS ---

def generate_graph(p):
    """Constructs a random undirected graph with NUM_NODES nodes."""
    return {i: {j for j in range(NUM_NODES) if j != i and random.random() < p} for i in range(NUM_NODES)}

def extract_k_core(graph, k):
    """Extracts the k-core from a graph."""
    while True:
        nodes_to_prune = [node for node, neighbors in graph.items() if len(neighbors) < k]
        if not nodes_to_prune:
            return graph
        for node in nodes_to_prune:
            for neighbor in list(graph[node]):  # Converting to list to avoid runtime modification issues
                if neighbor in graph:  # Checking if the neighbor still exists in the graph
                    graph[neighbor].discard(node)
            del graph[node]
    return graph


def determine_core_value(graph):
    """Computes the core value of a graph using binary search."""
    low, high, ans = 1, NUM_NODES, 0
    while low <= high:
        mid = (low + high) // 2
        if extract_k_core(graph.copy(), mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    return ans

def average_core(p):
    """Calculates the average core value over NUM_ITERATIONS for a given edge probability."""
    return sum(determine_core_value(generate_graph(p)) for _ in range(NUM_ITERATIONS)) / NUM_ITERATIONS

def visualize(graph, p):
    """Visualizes a graph using NetworkX."""
    k = determine_core_value(graph)
    core_graph = extract_k_core(graph, k)
    G = nx.Graph(core_graph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, with_labels=True, node_color='lightblue', node_size=1000, width=2.0, alpha=0.7)
    plt.title(f'Graph Visualization (p={p}, Core={k})')
    plt.show()

def plot_relation():
    """Plots the relationship between edge probability and core value."""
    probabilities = [i/100 for i in range(5, 100, 5)]
    core_values = [average_core(p) for p in probabilities]
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
    print(f"Sample: Average Core for p={sample_prob} is {average_core(sample_prob)}")
    for prob in [0.15, 0.5, 0.85]:
        graph = generate_graph(prob)
        visualize(graph, prob)
    plot_relation()

if __name__ == "__main__":
    main()
