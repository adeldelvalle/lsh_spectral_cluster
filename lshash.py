# lshash/lshash.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import int, round, str, object  # noqa

try:
    basestring
except NameError:
    basestring = str

# import future        # noqa
import builtins  # noqa
# import past          # noqa
import six  # noqa

import os
import json
import numpy as np

try:
    from storage import storage  # py2
except ImportError:
    from .storage import storage  # py3

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None

try:
    xrange  # py2
except NameError:
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
        return np.array(hashes_per_table).T  # (N, L) shape (points × tables)

    def find_strict_neighbors(self, points):
        edges = set()

        for point_idx, point in enumerate(points):
            neighbors = None

            for table_idx, table in enumerate(self.hash_tables):
                hash_val = self._hash_batch(self.uniform_planes[table_idx], np.array([point]))[0]
                bucket = set(table.get(hash_val, []))  # safer with .get() if key not present

                if neighbors is None:
                    neighbors = bucket
                else:
                    neighbors = neighbors.intersection(bucket)

                if not neighbors:
                    break  # early exit if no more candidates

            if neighbors:
                self.stitch(point_idx, neighbors, edges)

        return edges

    def stitch(self, point_idx, neighbors, edges):
        for neighbor in neighbors:
            edge = frozenset((point_idx, neighbor))
            edges.add(edge)

    def query(self, query_point, num_results=None, distance_func=None):
        """ Takes `query_point` which is either a tuple or a list of numbers,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.

        :param query_point:
            A list, or tuple, or numpy ndarray that only contains numbers.
            The dimension needs to be 1 * `input_dim`.
            Used by :meth:`._hash`.
        :param num_results:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
        :param distance_func:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        """

        candidates = set()
        if not distance_func:
            distance_func = "euclidean"

        if distance_func == "hamming":
            if not bitarray:
                raise ImportError(" Bitarray is required for hamming distance")

            for i, table in enumerate(self.hash_tables):
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                for key in table.keys():
                    distance = LSHash.hamming_dist(key, binary_hash)
                    if distance < 2:
                        candidates.update(table.get_list(key))

            d_func = LSHash.euclidean_dist_square

        else:

            if distance_func == "euclidean":
                d_func = LSHash.euclidean_dist_square
            elif distance_func == "true_euclidean":
                d_func = LSHash.euclidean_dist
            elif distance_func == "centred_euclidean":
                d_func = LSHash.euclidean_dist_centred
            elif distance_func == "cosine":
                d_func = LSHash.cosine_dist
            elif distance_func == "l1norm":
                d_func = LSHash.l1norm_dist
            else:
                raise ValueError("The distance function name is invalid.")

            for i, table in enumerate(self.hash_tables):
                binary_hash = self._hash(self.uniform_planes[i], query_point)
                candidates.update(table.get_list(binary_hash))

        # rank candidates by distance function
        candidates = [(ix, d_func(query_point, self._as_np_array(ix)))
                      for ix in candidates]
        candidates = sorted(candidates, key=lambda x: x[1])

        return candidates[:num_results] if num_results else candidates

    def get_hashes(self, input_point):
        """ Takes a single input point `input_point`, iterate through the
        uniform planes, and returns a list with size of `num_hashtables`
        containing the corresponding hash for each hashtable.

        :param input_point:
            A list, or tuple, or numpy ndarray object that contains numbers
            only. The dimension needs to be 1 * `input_dim`.
        """

        hashes = []
        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.uniform_planes[i], input_point)
            hashes.append(binary_hash)

        return hashes

    ### distance functions

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
