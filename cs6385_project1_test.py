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
    
    The graph is generated based on the Erdos-Renyi model, where each edge is
    included in the graph with probability p independent from every other edge.
    
    Args:
    - p (float): Probability of forming an edge between any two nodes.
    
    Returns:
    - dict: Graph represented as an adjacency list.
    """
    graph = {i: set() for i in range(25)}
    for i in range(25):
        for j in range(i+1, 25):
            if random.random() < p:
                graph[i].add(j)
                graph[j].add(i)
    return graph

def extract_k_core(graph, k):
    """
    Extracts the k-core from a graph.
    
    The k-core of a graph is a maximal subgraph in which each vertex has at least degree k.
    This function prunes nodes with degree less than k until all nodes in the graph satisfy this property.
    
    Args:
    - graph (dict): Input graph.
    - k (int): Desired core number.
    
    Returns:
    - dict: k-core of the graph.
    """
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

def determine_core_value(graph):
    """
    Computes the core value of a graph.
    
    The core value is the highest k for which a non-empty k-core exists in the graph.
    This function finds the largest k-core by iteratively checking and increasing k.
    
    Args:
    - graph (dict): Input graph.
    
    Returns:
    - int: Core value.
    """
    k = 1
    while extract_k_core(graph.copy(), k):
        k += 1
    return k - 1

def average_core(p):
    """
    Calculates the average core value over multiple iterations.
    
    For a given edge probability p, this function generates multiple random graphs
    and computes their average core value to provide a more stable estimate.
    
    Args:
    - p (float): Edge probability.
    
    Returns:
    - float: Average core value.
    """
    iterations = 10
    total_core = sum(determine_core_value(generate_graph(p)) for _ in range(iterations))
    return total_core / iterations

def visualize(p):
    """
    Visualizes a graph using NetworkX for a given edge probability.
    
    This function provides a visual representation of the graph structure,
    highlighting the nodes and their connections. It's useful for understanding
    the graph's topology and core structure.
    
    Args:
    - p (float): Edge probability.
    """
    graph = generate_graph(p)
    k = determine_core_value(graph)
    core_graph = extract_k_core(graph, k)
    G = nx.Graph(core_graph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, with_labels=True, node_color='lightblue', node_size=1000, width=2.0, alpha=0.7)
    plt.title(f'Graph Visualization (p={p}, Core={k})')
    plt.show()

def plot_relation():
    """
    Plots the relationship between edge probability and core value.
    
    This function visualizes how the core value of a graph changes as the edge probability varies.
    It provides insights into the stability and structure of the graph as the connectivity changes.
    """
    probabilities = [i/100 for i in range(5, 100, 5)]
    core_values = [average_core(p) for p in probabilities]
    plt.plot(probabilities, core_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('Edge Probability')
    plt.ylabel('Average Core Value')
    plt.title('Relationship between Edge Probability and Core Value')
    plt.grid(True)
    plt.show()

# --- TASK FUNCTIONS ---

def task1(p):
    """
    Task 1: Calculate average core value for a given edge probability.
    
    This task focuses on understanding the core structure of a graph for a specific edge probability.
    """
    return average_core(p)

def task2(p_values):
    """
    Task 2: Visualize the core numbers for a set of edge probabilities.
    
    Visualization helps in understanding the graph's structure and how the core varies with different probabilities.
    """
    for p in p_values:
        visualize(p)

def task3():
    """
    Task 3: Plot the relationship between edge probability and core value.
    
    This task provides a comprehensive view of how the core value changes across different edge probabilities.
    """
    plot_relation()

# --- MAIN EXECUTION ---

def main():
    """
    Main execution function.
    
    This function orchestrates the execution of all tasks, demonstrating the core concepts and visualizations.
    """
    sample_prob = 0.5
    print(f"Sample: Average Core for p={sample_prob} is {task1(sample_prob)}")
    task2([0.15, 0.5, 0.85])
    task3()

if __name__ == "__main__":
    main()
