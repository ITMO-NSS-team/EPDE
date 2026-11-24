"""Apply the experiment's host and numerical-library caps before importing NumPy."""
import ctypes
from ctypes import wintypes
import os


def cap_worker():
    import psutil
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        job = kernel.CreateJobObjectW(None, None)
        rate = (wintypes.DWORD * 2)(5, 200)
        if not (job and kernel.SetInformationJobObject(job, 15, ctypes.byref(rate), ctypes.sizeof(rate))
                and kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())):
            raise ctypes.WinError(ctypes.get_last_error())
    proc = psutil.Process()
    proc.cpu_affinity([proc.cpu_affinity()[-1]])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
