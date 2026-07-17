import numpy as np

from .utils import hadamard_matrix, get_mask
from .utils import validate_stats, visualize

from .sequency import get_matrix as get_sequency_matrix


def get_matrix(n):
    return hadamard_matrix(n)


def get_index_matrix(n):
    H = get_sequency_matrix(n)
    Hbig = get_matrix(n**2)
    index_matrix = [
        (H @ h.reshape(n, n) @ H.T > 0) * (n**2 - i) for i, h in enumerate(Hbig)
    ]
    index_matrix = np.stack(index_matrix, axis=0).sum(axis=0)

    return index_matrix.astype(np.int32)
