from study_context import *
import ctypes
ORIGINAL=F.State.axis_score
lib=ctypes.CDLL(str(ROOT/'native_score.dll'));ptr=ctypes.POINTER(ctypes.c_double)
call=lib.innovations;call.argtypes=[ptr,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ptr,ptr,ctypes.c_void_p];call.restype=ctypes.c_int
crt=ctypes.CDLL('ucrtbase.dll');logptr=ctypes.cast(crt.log,ctypes.c_void_p)

def axis_score(data,r,k,p1,p2):
    if data.shape[1]!=1:return ORIGINAL(data,r,k,p1,p2)
    data=np.ascontiguousarray(data,dtype=np.float64);out=np.zeros(10);workspace=np.zeros(1800)
    status=call(data.ctypes.data_as(ptr),len(data),r,k,p1,p2,out.ctypes.data_as(ptr),workspace.ctypes.data_as(ptr),logptr)
    if status:raise FloatingPointError('Nonpositive native innovations variance')
    return float(out[0]),float(out[1]),out[2:6],out[6:10]

def install():F.State.axis_score=axis_score
def restore():F.State.axis_score=ORIGINAL
