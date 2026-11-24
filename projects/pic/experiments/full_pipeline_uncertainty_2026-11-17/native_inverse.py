"""Optional runtime probe; never imported by the primary experiment."""
from bridge import *
import ctypes

DLL = ROOT / 'native_probe' / 'selected_inverse.dll'
lib = ctypes.CDLL(str(DLL))
call = lib.selected_inverse
call.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                 ctypes.c_size_t, ctypes.c_size_t,
                 ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
call.restype = ctypes.c_int


def selected_inverse(chol):
    chol = np.ascontiguousarray(chol, dtype=np.float64)
    width, n = chol.shape
    inv = np.zeros_like(chol)
    weights, off = np.empty(width), np.empty(width)
    ptr = ctypes.POINTER(ctypes.c_double)
    code = call(chol.ctypes.data_as(ptr), inv.ctypes.data_as(ptr), n, width-1,
                weights.ctypes.data_as(ptr), off.ctypes.data_as(ptr))
    if code:
        raise MemoryError('Native selected inverse allocation failed')
    return inv
