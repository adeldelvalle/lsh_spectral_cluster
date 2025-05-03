from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchvision import datasets, transforms
import torch
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import numpy as np
from lshash import LSHash
import igraph as ig

import leidenalg

#
# # Load Fashion-MNIST
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
#
points = fashion_mnist['data']  # (70000, 784)
labels = fashion_mnist['target']  # String labels: '0', '1', ..., '9'
#
# # Convert labels to integers
labels = labels.astype(int)
def get_cifar10_numpy(flatten=True, subset=10000):
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=subset, shuffle=True)
    images, labels = next(iter(loader))

    if flatten:
        images = images.view(images.size(0), -1)

    return images.numpy(), labels.numpy()


#points, labels = get_cifar10_numpy()

# Load the small Digits dataset (8x8 images)
# from sklearn.datasets import fetch_openml
#
# mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#
# points = mnist.data
# labels = mnist.target.astype(int)
#
# # Optional: take only 60,000 if you want train only
points = points[:10000]
labels = labels[:10000]

# PCA for visualization
pca = PCA(n_components=2)
points_2d = pca.fit_transform(points)

# Plot
plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, cmap='tab10', s=15)
plt.title("Digits dataset (PCA projection)")
plt.colorbar(label='Digit label')
plt.show()

lsh = LSHash(10, points.shape[1], 35)
lsh.index_batch(points)
edges = lsh.find_topk_neighbors_with_weights(points)

G = nx.Graph()
G.add_weighted_edges_from(edges)
#
# # Add nodes
G.add_nodes_from(range(len(points)))  # Add all points as nodes
#
# # Add edges
for edge in edges:
    node_list = list(edge)
    G.add_edge(node_list[0], node_list[1])
#
# # Build the pos dictionary for plotting
pos = {i: points_2d[i] for i in range(len(points))}
#
import community as community_louvain  # this is the Louvain package

# Run Louvain clustering
partition = community_louvain.best_partition(G, weight='weight')

# 'partition' is a dict: node index -> community id
# Example: {0: 2, 1: 1, 2: 2, 3: 0, ...}
degrees = [deg for _, deg in G.degree()]
plt.hist(degrees, bins=50)
# Extract community labels
louvain_labels = np.array([partition.get(i, -1) for i in range(len(points))])

from scipy.stats import mode

# Match Louvain clusters to majority digit labels
label_mapping = {}
unique_louvain_clusters = np.unique(louvain_labels)

for cluster_id in unique_louvain_clusters:
    mask = (louvain_labels == cluster_id)
    majority_label = mode(labels[mask], keepdims=False).mode
    label_mapping[cluster_id] = majority_label

# Remap Louvain labels
mapped_labels = np.array([label_mapping[cluster_id] for cluster_id in louvain_labels])

# Normalize Louvain labels between 0 and 1
norm_labels = (louvain_labels - np.min(louvain_labels)) / (np.max(louvain_labels) - np.min(louvain_labels))
#
# # Use a continuous colormap like 'viridis' or 'plasma'
cmap = cm.get_cmap('nipy_spectral')

plt.figure(figsize=(8, 6))
plt.scatter(points_2d[:, 0], points_2d[:, 1], c=norm_labels, cmap=cmap, s=20)
plt.title("PCA Projection with Louvain Communities (continuous colors)")
plt.colorbar(label='Cluster ID (normalized)')
plt.show()

# Plot graph with Louvain communities
plt.figure(figsize=(8, 6))
nx.draw(
    G, pos, node_size=20,
    node_color=louvain_labels, cmap=cmap,
    with_labels=False, edge_color='gray'
)
plt.title("Louvain Clustering on Stitched Graph (PCA projection)")
plt.show()

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Compute scores between ground truth labels and your mapped Louvain labels
ari_score = adjusted_rand_score(labels, louvain_labels)
nmi_score = normalized_mutual_info_score(labels, louvain_labels)

print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Number of Louvain communities: {len(np.unique(louvain_labels))}")
# from collections import Counter
#
# counter = Counter(louvain_labels)
# print(f"Cluster sizes (top 10 largest): {counter.most_common(10)}")
#
# # Graph layout based on edge forces (not PCA)
# pos_spring = nx.spring_layout(G, iterations=30)
#
# plt.figure(figsize=(8, 6))
# nx.draw(
#     G, pos_spring, node_size=20,
#     node_color=louvain_labels, cmap=cmap,
#     with_labels=False, edge_color='gray'
# )
# plt.title("Louvain Clustering on Stitched Graph (Spring Layout)")
# plt.show()

# Convert NetworkX graph to iGraph
G_ig = ig.Graph()
G_ig.add_vertices(len(G.nodes))
edges_ig = [(u, v) for u, v in G.edges()]
weights = [G[u][v].get('weight', 1.0) for u, v in edges_ig]

G_ig.add_edges(edges_ig)
G_ig.es['weight'] = weights

# Run Leiden community detection
partition = leidenalg.find_partition(
    G_ig,
    leidenalg.ModularityVertexPartition,  # or CPMVertexPartition for fine-tuned control
    weights=G_ig.es['weight']
)

# Map Leiden labels to node IDs (igraph uses indices)
leiden_labels = np.array(partition.membership)  # index: node, value: cluster ID

ari_score = adjusted_rand_score(labels, leiden_labels)
nmi_score = normalized_mutual_info_score(labels, leiden_labels)

print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Number of Leiden communities: {len(np.unique(leiden_labels))}")
norm_labels = (leiden_labels - np.min(leiden_labels)) / (np.max(leiden_labels) - np.min(leiden_labels))

plt.figure(figsize=(8, 6))
plt.scatter(points_2d[:, 0], points_2d[:, 1], c=norm_labels, cmap=cmap, s=20)
plt.title("PCA Projection with Louvain Communities (continuous colors)")
plt.colorbar(label='Cluster ID (normalized)')
plt.show()
