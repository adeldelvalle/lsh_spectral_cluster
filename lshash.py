# lshash/lshash.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import int, round, str, object  # noqa
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

# import future        # noqa
import builtins  # noqa
# import past          # noqa
import six  # noqa

import os
import json
import numpy as np
from sklearn.decomposition import PCA

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None

xrange = range  # py3


class LSHash(object):
    """ LSHash implments locality sensitive hashing using random projection for
    input vectors of dimension `input_dim`.

    Attributes:

    :param hash_size:
        The length of the resulting binary hash in integer. E.g., 32 means the
        resulting binary hash will be 32-bit long.
    :param input_dim:
        The dimension of the input vector. E.g., a grey-scale picture of 30x30
        pixels will have an input dimension of 900.
    :param num_hashtables:
        (optional) The number of hash tables used for multiple lookups.
    :param storage_config:
        (optional) A dictionary of the form `{backend_name: config}` where
        `backend_name` is the either `dict` or `redis`, and `config` is the
        configuration used by the backend. For `redis` it should be in the
        format of `{"redis": {"host": hostname, "port": port_num}}`, where
        `hostname` is normally `localhost` and `port` is normally 6379.
    :param matrices_filename:
        (optional) Specify the path to the compressed numpy file ending with
        extension `.npz`, where the uniform random planes are stored, or to be
        stored if the file does not exist yet.
    :param overwrite:
        (optional) Whether to overwrite the matrices file if it already exist
    """

    def __init__(self, hash_size, input_dim, num_hashtables=1,
                 storage_config=None, matrices_filename=None, hashtable_filename=None, overwrite=False):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        self.overwrite = overwrite

        self._init_uniform_planes()
        self._init_hashtables()

    def _init_uniform_planes(self):
        """ Initialize uniform planes used to calculate the hashes

        if file `self.matrices_filename` exist and `self.overwrite` is
        selected, save the uniform planes to the specified file.

        if file `self.matrices_filename` exist and `self.overwrite` is not
        selected, load the matrix with `np.load`.

        if file `self.matrices_filename` does not exist and regardless of
        `self.overwrite`, only set `self.uniform_planes`.
        """

        self.uniform_planes = [self._generate_uniform_planes()
                               for _ in xrange(self.num_hashtables)]

    def _init_hashtables(self):
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        self.hash_tables = [{} for i in xrange(self.num_hashtables)]

    def _generate_uniform_planes(self):
        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """

        return np.random.randn(self.hash_size, self.input_dim)

    def _hash_batch(self, planes, input_points):
        """
        Vectorized hashing for a batch of input points.

        :param planes: numpy array of shape (hash_size, input_dim)
        :param input_points: numpy array of shape (N, input_dim)
        :returns: list of binary hash strings for each input point
        """
        try:
            input_points = np.array(input_points)  # (N, input_dim)
            projections = np.dot(input_points, planes.T)  # (N, hash_size)
        except TypeError as e:
            print("The input points must be an array-like object with numbers only.")
            raise
        except ValueError as e:
            print("Dimension mismatch between input points and planes.", e)
            raise
        else:
            # projections > 0 → 1, else 0
            binary_matrix = (projections > 0).astype(int)  # (N, hash_size)

            # Convert each row to a binary string
            hash_strings = ["".join(map(str, row)) for row in binary_matrix]

            return hash_strings

    def index_batch(self, input_points):
        """
        Index a batch of input points at once.

        :param input_points: numpy array of shape (N, input_dim)
        """
        hashes_per_table = []

        for i, table in enumerate(self.hash_tables):
            # Vectorized hash computation for all points in table i
            hashes = self._hash_batch(self.uniform_planes[i], input_points)  # (N,) list of hash strings

            # Append each point_idx to its corresponding hash bucket
            for point_idx, h in enumerate(hashes):
                if h not in table:
                    table[h] = []
                table[h].append(point_idx)

            hashes_per_table.append(hashes)

        # hashes_per_table is a list of L lists, each containing N hash strings
        # return np.array(hashes_per_table).T  # (N, L) shape (points × tables)

    def find_topk_neighbors_with_weights(self, points, k=34):
        """
        Return a list of weighted edges based on LSH voting, keeping only top-k neighbors per point.
        Each edge is (point_i, point_j, weight), where weight is number of shared buckets.
        """
        edge_weights = dict()

        for point_idx, point in enumerate(points):
            vote_counter = dict()

            # Count how many times each point appeared in the same bucket as this point
            for table_idx, table in enumerate(self.hash_tables):
                hash_val = self._hash_batch(self.uniform_planes[table_idx], np.array([point]))[0]
                bucket = set(table.get(hash_val, []))

                for neighbor_idx in bucket:
                    if neighbor_idx == point_idx:
                        continue
                    vote_counter[neighbor_idx] = vote_counter.get(neighbor_idx, 0) + 1

            # Select top-k neighbors with highest votes
            top_neighbors = sorted(vote_counter.items(), key=lambda x: -x[1])[:k]

            for neighbor_idx, votes in top_neighbors:
                edge = tuple(sorted((point_idx, neighbor_idx)))
                if edge not in edge_weights or votes > edge_weights[edge]:
                    edge_weights[edge] = votes

        # Convert to list of weighted edges
        weighted_edges = [(i, j, w) for (i, j), w in edge_weights.items()]
        return weighted_edges




from sklearn.datasets import load_digits

from sklearn.datasets import fetch_openml
import numpy as np

# Load Fashion-MNIST
#fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

#points = fashion_mnist['data']  # (70000, 784)
#labels = fashion_mnist['target']  # String labels: '0', '1', ..., '9'

# Convert labels to integers
#labels = labels.astype(int)

# Load the small Digits dataset (8x8 images)
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

points = mnist.data
labels = mnist.target.astype(int)

# Optional: take only 60,000 if you want train only
points = points[:50000]
labels = labels[:50000]


# PCA for visualization
pca = PCA(n_components=2)
points_2d = pca.fit_transform(points)

# Plot
plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, cmap='tab10', s=15)
plt.title("Digits dataset (PCA projection)")
plt.colorbar(label='Digit label')
plt.show()

lsh = LSHash(10, points.shape[1], 40)
lsh.index_batch(points)
edges = lsh.find_topk_neighbors_with_weights(points)

G = nx.Graph()
G.add_weighted_edges_from(edges)


# Add nodes
G.add_nodes_from(range(len(points)))  # Add all points as nodes

# Add edges
for edge in edges:
    node_list = list(edge)
    G.add_edge(node_list[0], node_list[1])


# Build the pos dictionary for plotting
pos = {i: points_2d[i] for i in range(len(points))}


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

# Use a continuous colormap like 'viridis' or 'plasma'
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
from collections import Counter
counter = Counter(louvain_labels)
print(f"Cluster sizes (top 10 largest): {counter.most_common(10)}")

# Graph layout based on edge forces (not PCA)
pos_spring = nx.spring_layout(G, iterations=30)

plt.figure(figsize=(8, 6))
nx.draw(
    G, pos_spring, node_size=20,
    node_color=louvain_labels, cmap=cmap,
    with_labels=False, edge_color='gray'
)
plt.title("Louvain Clustering on Stitched Graph (Spring Layout)")
plt.show()



