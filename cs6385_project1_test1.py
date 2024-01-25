#Revanth Chenna
#RXC210069
#CS6385_Project1

import random
import matplotlib.pyplot as plt
import networkx as nx

# --- UTILITY FUNCTIONS ---

def generate_graph(p):
    """
    Constructs a random undirected graph with 25 nodes.
    
    Args:
    - p (float): Probability of forming an edge between any two nodes.
    
    Returns:
    - dict: Graph represented as an adjacency list.
    
    The function initializes an empty graph with 25 nodes. For each pair of nodes,
    it checks against a random number to decide if an edge should be formed based on probability p.
    """
    graph = {i: set() for i in range(25)}
    for i in range(25):
        for j in range(i+1, 25):
            # If random number is less than p, form an edge
            if random.random() < p:
                graph[i].add(j)
                graph[j].add(i)
    return graph

def extract_k_core(graph, k):
    """
    Extracts the k-core from a graph. The k-core is the largest subgraph where each node has at least k neighbors.
    
    Args:
    - graph (dict): Input graph.
    - k (int): Desired core number.
    
    Returns:
    - dict: k-core of the graph.
    
    The function iteratively prunes nodes with fewer than k neighbors until no such nodes remain.
    """
    while True:
        nodes_to_prune = [node for node, neighbors in graph.items() if len(neighbors) < k]
        if not nodes_to_prune:
            break
        for node in nodes_to_prune:
            for neighbor in list(graph[node]):  # Converting to list to avoid runtime modification issues
                if neighbor in graph:  # Checking if the neighbor still exists in the graph
                    graph[neighbor].discard(node)
            del graph[node]
    return graph

def determine_core_value(graph):
    """
    Computes the core value of a graph, which is the highest k for which a non-empty k-core exists.
    
    Args:
    - graph (dict): Input graph.
    
    Returns:
    - int: Core value.
    
    The function iteratively checks for the existence of k-cores, increasing k until no k-core is found.
    """
    k = 1
    while extract_k_core(graph.copy(), k):
        k += 1
    return k - 1

def average_core(p):
    """
    Calculates the average core value over multiple iterations for a given edge probability.
    
    Args:
    - p (float): Edge probability.
    
    Returns:
    - float: Average core value.
    
    The function generates multiple random graphs for the given edge probability and computes their average core value.
    """
    iterations = 10
    total_core = sum(determine_core_value(generate_graph(p)) for _ in range(iterations))
    return total_core / iterations

def visualize(p):
    """
    Visualizes a graph using NetworkX for a given edge probability.
    
    Args:
    - p (float): Edge probability.
    
    The function generates a random graph, computes its core value, extracts the core subgraph, and then visualizes it.
    """
    
    # Generate a random graph based on the given edge probability p
    graph = generate_graph(p)
    
    # Compute the core value (k) of the generated graph
    k = determine_core_value(graph)
    
    # Extract the k-core subgraph from the original graph
    core_graph = extract_k_core(graph, k)
    
    # Convert the core graph dictionary into a NetworkX graph object
    G = nx.Graph(core_graph)
    
    # Determine the layout for the graph visualization (spring layout is force-directed)
    layout = nx.spring_layout(G)
    
    # Draw the graph using the specified layout, node colors, sizes, and other properties
    nx.draw(G, layout, with_labels=True, node_color='lightblue', node_size=1000, width=2.0, alpha=0.7)
    
    # Set the title for the graph visualization
    plt.title(f'Graph Visualization (p={p}, Core={k})')
    
    # Display the graph visualization
    plt.show()


def plot_relation():
    """
    Plots the relationship between edge probability and core value.
    
    The function computes average core values for a range of edge probabilities and plots the results.
    """
    # Define a list of edge probabilities ranging from 0.05 to 0.95 in increments of 0.05
    probabilities = [i/100 for i in range(5, 100, 5)]
    
    # Compute the average core values for each edge probability
    core_values = [average_core(p) for p in probabilities]
    
    # Plot the relationship between edge probabilities and their corresponding average core values
    plt.plot(probabilities, core_values, marker='o', linestyle='-', color='blue')
    
    # Set the x-axis label
    plt.xlabel('Edge Probability')
    
    # Set the y-axis label
    plt.ylabel('Average Core Value')
    
    # Set the title of the plot
    plt.title('Relationship between Edge Probability and Core Value')
    
    # Display a grid on the plot for better readability
    plt.grid(True)
    
    # Show the plot
    plt.show()


# --- MAIN EXECUTION ---

def main():
    """
    Main execution function.
    
    Demonstrates the functionality by:
    1. Printing the average core value for a sample edge probability.
    2. Visualizing graphs for a set of edge probabilities.
    3. Plotting the relationship between edge probability and core value.
    """
    sample_prob = 0.5
    print(f"Sample: Average Core for p={sample_prob} is {average_core(sample_prob)}")
    
    for prob in [0.15, 0.5, 0.85]:
        visualize(prob)
    
    plot_relation()

if __name__ == "__main__":
    main()
