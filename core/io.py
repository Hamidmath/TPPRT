"""
Loader for popularity / smoothed .npz files.

The raw popularity matrix is genuinely sparse and stored as CSR.
The smoothed matrix is dense after Laplace smoothing, so saving it as
CSR is counterproductive (doubles memory and disk). Modern runs save
it dense under the key `matrix`; legacy runs saved it as CSR under
`matrix_data/indices/indptr/shape`.

`load_popularity_npz(path)` detects either layout and returns an
object with a uniform `.getrow(i)` API (scipy CSR row, tiny) and a
`.shape` attribute, so existing callers written against `csr_matrix`
continue to work unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class _DenseCSR:
    """Dense-backed matrix with a minimal CSR-like API."""

    def __init__(self, dense: np.ndarray):
        self._d = dense
        self.shape = dense.shape

    def getrow(self, i: int) -> csr_matrix:
        # One-row CSR view of a single dense row (cheap, ~N*4 bytes).
        return csr_matrix(self._d[i : i + 1])

    def toarray(self) -> np.ndarray:
        return self._d


def load_popularity_npz(path: str):
    """Load a popularity .npz produced by analyze_popularity.py or
    generate_smoothed.py. Returns a dict with keys:
        matrix    -- CSR-compatible matrix (supports .getrow and .shape)
        times     -- list of bin-start timestamp strings
        link_ids  -- list of link-id strings (column labels)
    """
    loader = np.load(path, allow_pickle=True)
    if 'matrix' in loader.files:
        # Dense layout (new smoothed output).
        matrix = _DenseCSR(loader['matrix'])
    else:
        # Legacy CSR layout (raw popularity file, and old smoothed files).
        matrix = csr_matrix(
            (loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']),
            shape=tuple(loader['matrix_shape']),
        )
    return {
        'matrix': matrix,
        'times': list(loader['times']),
        'link_ids': list(loader['link_ids']),
    }
