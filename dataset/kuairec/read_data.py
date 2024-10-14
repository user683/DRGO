import dgl
import numpy as np
import torch
import networkx as nx
from joblib import Parallel, delayed

# Load dataset
dataset_paths = {"train": "kuairec_train_data.bin",
                 "test": "kuairec_test_data.bin", }

datasets = {key: dgl.load_graphs(path)[0][0] for key, path in dataset_paths.items()}

train_graph = datasets['train']

# Extract edge indices
edge_index = torch.stack(train_graph.edges())
print(edge_index)

np.save("edge_index.npy", edge_index)

import numpy as np
import networkx as nx
import time


def format_time(seconds):
    """Format seconds into Hours:Minutes:Seconds"""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def softmax(x):
    """Apply the Softmax function to the input array x"""
    e_x = np.exp(x - np.max(x))  # Subtract the maximum value for numerical stability
    return e_x / e_x.sum()


# Record the start time
print("Starting calculation of betweenness centrality")
start_time = time.time()

# Load edge index data
edge_index = np.load("edge_index.npy")

# Build the graph
G = nx.Graph()
edges = list(zip(edge_index[0], edge_index[1]))
G.add_edges_from(edges)

# Calculate betweenness centrality
centrality = nx.betweenness_centrality(G, k=1000, normalized=True)

# Apply Softmax normalization to the centrality results
centrality_values = np.array(list(centrality.values()))
centrality_softmax = softmax(centrality_values)

# Save the normalized results as a dictionary, formatted as {user_id: softmax_value}
centrality_dict_softmax = {node: value for node, value in zip(centrality.keys(), centrality_softmax)}

# Save the result as a .npy file
np.save("centrality_dict_softmax.npy", centrality_dict_softmax)

# Record the end time
end_time = time.time()

print(f"Betweenness centrality calculation completed and normalized with Softmax, time taken: {format_time(end_time - start_time)}")
