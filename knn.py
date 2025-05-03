import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import community as community_louvain  # this is python-louvain
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def knn_graph_louvain_clustering(X, k=10):
    """
    Build a kNN graph from data X and cluster with Louvain algorithm.
    Returns: predicted_labels, Louvain partition dictionary
    """
    # Step 1: Build kNN graph
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    G = nx.Graph()
    for i in range(len(X)):
        for j in indices[i][1:]:  # skip self
            G.add_edge(i, j, weight=1)  # can use weight=1/(1+distance) if desired

    # Step 2: Apply Louvain clustering
    partition = community_louvain.best_partition(G, weight='weight')

    # Convert partition (dict) to label array
    predicted_labels = np.array([partition[i] for i in range(len(X))])
    return predicted_labels, partition

# Example usage
# X = your dataset (e.g. X = fashion_mnist_10k as numpy array)
# y_true = true labels (optional, for ARI/NMI)

# predicted_labels, _ = knn_graph_louvain_clustering(X, k=10)

# Optional metrics:
# print("ARI:", adjusted_rand_score(y_true, predicted_labels))
# print("NMI:", normalized_mutual_info_score(y_true, predicted_labels))
