from .hadamard import (
    get_matrix as hadamard_matrix,
    get_index_matrix as hadamard_index_matrix,
)
from .sequency import (
    get_matrix as sequency_matrix,
    get_index_matrix as sequency_index_matrix,
)
from .cake_cutting import (
    get_matrix as cake_cutting_matrix,
    get_index_matrix as cake_cutting_index_matrix,
)
from .zig_zag import (
    get_matrix as zig_zag_matrix,
    get_index_matrix as zig_zag_index_matrix,
)
from .xy import get_matrix as xy_matrix, get_index_matrix as xy_index_matrix

from .utils import get_mask


MATRIX_FUNCTIONS = {
    "hadamard": hadamard_matrix,
    "sequency": sequency_matrix,
    "cake_cutting": cake_cutting_matrix,
    "zig_zag": zig_zag_matrix,
    "XY": xy_matrix,
}

INDEX_MATRIX_FUNCTIONS = {
    "hadamard": hadamard_index_matrix,
    "sequency": sequency_index_matrix,
    "cake_cutting": cake_cutting_index_matrix,
    "zig_zag": zig_zag_index_matrix,
    "XY": xy_index_matrix,
}


def get_matrix(n, ordering="hadamard"):
    return MATRIX_FUNCTIONS[ordering](n)


def get_index_matrix(n, ordering="hadamard"):
    return INDEX_MATRIX_FUNCTIONS[ordering](n)


def get_n_mask(size, n, ordering="hadamard"):
    index_matrix = get_index_matrix(size, ordering)
    mask = get_mask(index_matrix, size, n)
    return mask
