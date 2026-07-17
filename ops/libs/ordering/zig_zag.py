import numpy as np

from .utils import get_order, get_coords_array, coord_ordering
from .utils import get_mask
from .utils import validate_stats, visualize

from .sequency import get_matrix as hadamard_matrix


def zigzag_order(n):
    return coord_ordering(n, get_index_matrix)


def get_matrix(n):
    size = np.sqrt(n).astype(np.int32)
    _, coords = zigzag_order(size)

    Hs = hadamard_matrix(size)
    H = [np.outer(Hs[i], Hs[j]) for i, j in zip(coords[:, 0], coords[:, 1])]
    H = np.stack(H, 0).reshape(n, -1)
    return H


def get_index_matrix(n):
    index_matrix = get_coords_array(n)
    index_matrix = index_matrix.sum(-1)
    index_matrix = index_matrix / index_matrix.max()
    index_matrix = 1 - index_matrix
    index_matrix = get_order(index_matrix)
    return index_matrix
