# lshash/lshash.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import int, round, str, object  # noqa
# import future        # noqa
import builtins  # noqa
# import past          # noqa
import six  # noqa
import math
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

    def _generate_uniform_planes(self, similarity_threshold=0.8, max_attempts=5000):
        """
        Generate a 2D numpy array of quasi-orthogonal projection vectors.
        Each row is a normalized projection vector.
        """
        planes = []
        attempts = 0

        while len(planes) < self.hash_size and attempts < max_attempts:
            v = np.random.randn(self.input_dim)
            v /= np.linalg.norm(v)

            is_similar = any(
                abs(np.dot(v, u)) > similarity_threshold for u in planes
            )

            if not is_similar:
                planes.append(v)

            attempts += 1

        if len(planes) < self.hash_size:
            print(f"⚠️ Only generated {len(planes)} diverse planes out of requested {self.hash_size}")

        return np.array(planes)

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

    def find_topk_neighbors_with_weights(self, points, k=15):
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
                    vote_counter[neighbor_idx] = vote_counter.get(neighbor_idx, 0) + (1 / math.log(len(bucket) + 1))

            # Select top-k neighbors with highest votes
            top_neighbors = sorted(vote_counter.items(), key=lambda x: -x[1])[:k]

            for neighbor_idx, votes in top_neighbors:
                edge = tuple(sorted((point_idx, neighbor_idx)))
                if edge not in edge_weights:
                    edge_weights[edge] = np.exp(votes/k)
                else:
                    edge_weights[edge] = edge_weights[edge]**2



        # Convert to list of weighted edges
        weighted_edges = [(i, j, w) for (i, j), w in edge_weights.items()]
        return weighted_edges


