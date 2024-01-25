import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Data Structures & Utility Functions

def generate_random_graph(p):
    graph = {i: set() for i in range(25)}
    for i in range(25):
        for j in range(i+1, 25):
            if random.random() < p:
                graph[i].add(j)
                graph[j].add(i)
    return graph

def compute_k_core(graph, k):
    while True:
        nodes_to_remove = [node for node, neighbors in graph.items() if len(neighbors) < k]
        if not nodes_to_remove:
            break
        for node in nodes_to_remove:
            del graph[node]
        for remaining_node in graph:
            graph[remaining_node] = graph[remaining_node] - set(nodes_to_remove)
    return graph

def core_number(graph):
    k = 1
    while True:
        core = compute_k_core(graph.copy(), k)
        if not core:
            return k - 1
        k += 1
        if not any(core.values()):  # Check if the core is empty
            return k - 1

# Task 1

def compute_average_core(p):
    total_core = 0
    for _ in range(10):
        graph = generate_random_graph(p)
        total_core += core_number(graph)
    return total_core / 10

# Task 2: Enhanced Visualization using Plotly

def visualize_graph_with_plotly(p):
    graph = generate_random_graph(p)
    k = core_number(graph)
    core = compute_k_core(graph, k)
    
    edge_x = []
    edge_y = []
    for node, neighbors in core.items():
        for neighbor in neighbors:
            x0, y0 = node % 5, node // 5
            x1, y1 = neighbor % 5, neighbor // 5
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = [i % 5 for i in core.keys()]
    node_y = [i // 5 for i in core.keys()]
    node_text = [str(i) for i in core.keys()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_text,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        textposition="bottom center")
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=f'Graph for p={p} with Core={k}',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()

# Task 3

def plot_core_vs_p():
    p_values = [i/100 for i in range(5, 100, 5)]
    core_values = [compute_average_core(p) for p in p_values]
    
    plt.plot(p_values, core_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Edge Probability p')
    plt.ylabel('Average Core Value')
    plt.title('Core(p) vs p')
    plt.grid(True)
    plt.show()

# Automatic Execution

def main():
    # Task 1: Display average core for a sample p value
    p_sample = 0.5
    print(f"Average Core for p={p_sample}: {compute_average_core(p_sample)}")
    
    # Task 2: Visualize graphs for three different p values
    for p in [0.15, 0.5, 0.85]:
        visualize_graph_with_plotly(p)
    
    # Task 3: Plot Core(p) vs p
    plot_core_vs_p()

if __name__ == "__main__":
    main()
