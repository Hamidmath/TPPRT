"""
GPU-accelerated sparse power iteration for Two-Phase PageRank.

Backends (tried in order):
  1. CuPy + cuSPARSE  (fastest, needs CUDA toolkit)
  2. PyTorch sparse   (good, ships its own CUDA libs)
  3. SciPy CPU        (always works)

Installation for GPU:
  Option A (CuPy):  sudo apt install nvidia-cuda-toolkit && pip install cupy-cuda12x
  Option B (Torch):  pip install torch --index-url https://download.pytorch.org/whl/cu121

Usage:
  from core.gpu_backend import power_iteration, BACKEND_NAME
  v = power_iteration(M_csr, E_2N, damping=0.96)
"""

import numpy as np
from scipy.sparse import csr_matrix

BACKEND_NAME = "cpu"

# ── Try CuPy ────────────────────────────────────────────────────

_cupy_ok = False
try:
    import cupy as _cp
    import cupyx.scipy.sparse as _cps
    # Smoke-test: actually create a sparse matrix and do matvec
    _test_m = _cps.csr_matrix(csr_matrix(np.eye(2)))
    _test_v = _cp.array([1.0, 2.0])
    _ = _test_m.T.dot(_test_v)
    _cp.cuda.Stream.null.synchronize()
    _cupy_ok = True
    BACKEND_NAME = "cupy"
except Exception:
    _cupy_ok = False

# ── Try PyTorch ──────────────────────────────────────────────────

_torch_ok = False
if not _cupy_ok:
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            # Smoke-test sparse matvec on GPU
            _i = _torch.tensor([[0, 1], [0, 1]], dtype=_torch.long)
            _v = _torch.tensor([1.0, 1.0], dtype=_torch.float64)
            _s = _torch.sparse_coo_tensor(_i, _v, (2, 2)).to('cuda').to_sparse_csr()
            _d = _torch.tensor([1.0, 2.0], dtype=_torch.float64, device='cuda')
            _ = _torch.mv(_s, _d)
            _torch.cuda.synchronize()
            _torch_ok = True
            BACKEND_NAME = "torch_gpu"
    except Exception:
        _torch_ok = False

print(f"[gpu_backend] Active backend: {BACKEND_NAME}")


# ── Power iteration implementations ─────────────────────────────

def _pi_cupy(M_csr, E_2N, damping, tol, max_iter):
    """CuPy GPU power iteration."""
    M_gpu = _cps.csr_matrix(M_csr)
    E_gpu = _cp.asarray(E_2N)
    v = E_gpu.copy()
    for k in range(max_iter):
        v_new = damping * M_gpu.T.dot(v) + (1.0 - damping) * E_gpu
        diff = float(_cp.sum(_cp.abs(v_new - v)))
        v = v_new
        if diff < tol:
            break
    return _cp.asnumpy(v)


def _pi_torch(M_csr, E_2N, damping, tol, max_iter):
    """PyTorch GPU power iteration."""
    device = _torch.device('cuda')
    # Convert scipy CSR to torch sparse CSR
    M_csr = M_csr.tocsr()
    crow = _torch.tensor(M_csr.indptr, dtype=_torch.long, device=device)
    col = _torch.tensor(M_csr.indices, dtype=_torch.long, device=device)
    val = _torch.tensor(M_csr.data, dtype=_torch.float64, device=device)
    Mt = _torch.sparse_csr_tensor(crow, col, val, M_csr.shape,
                                   device=device).t().to_sparse_csr()

    E_gpu = _torch.tensor(E_2N, dtype=_torch.float64, device=device)
    v = E_gpu.clone()
    for k in range(max_iter):
        v_new = damping * _torch.mv(Mt, v) + (1.0 - damping) * E_gpu
        diff = float(_torch.sum(_torch.abs(v_new - v)))
        v = v_new
        if diff < tol:
            break
    return v.cpu().numpy()


def _pi_cpu(M_csr, E_2N, damping, tol, max_iter):
    """SciPy CPU power iteration."""
    v = E_2N.copy()
    for k in range(max_iter):
        v_new = damping * M_csr.T.dot(v) + (1.0 - damping) * E_2N
        diff = np.sum(np.abs(v_new - v))
        v = v_new
        if diff < tol:
            break
    return v


def power_iteration(M_csr, E_2N, damping=0.80, tol=1e-6, max_iter=100):
    """Run power iteration on the best available backend.

    Args:
        M_csr: scipy.sparse.csr_matrix (2N x 2N transition matrix)
        E_2N: numpy array (2N teleportation vector)
        damping: float (PageRank damping factor)
        tol: float (L1 convergence tolerance)
        max_iter: int (maximum iterations)

    Returns:
        numpy array (2N steady-state vector)
    """
    if _cupy_ok:
        return _pi_cupy(M_csr, E_2N, damping, tol, max_iter)
    elif _torch_ok:
        return _pi_torch(M_csr, E_2N, damping, tol, max_iter)
    else:
        return _pi_cpu(M_csr, E_2N, damping, tol, max_iter)
