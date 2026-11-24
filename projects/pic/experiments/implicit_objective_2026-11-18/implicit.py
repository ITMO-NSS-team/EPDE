"""Real homogeneous relations in a weak moment matrix."""
import numpy as np


def fit(matrix):
    x = np.vstack((matrix.real, matrix.imag)) if np.iscomplexobj(matrix) else matrix
    scale = np.linalg.norm(x, axis=0)
    if np.any(scale == 0):
        return None, 'zero training column'
    normalized = x / scale
    rows, cols = x.shape
    _, singular, right = np.linalg.svd(normalized,full_matrices=rows<cols)
    singular = np.pad(singular,(0,cols-len(singular)))
    tolerance = 64*np.finfo(float).eps*max(rows,cols)*singular[0]
    if cols>1 and singular[-2]-singular[-1] <= tolerance:
        return None, 'nonunique direction'
    return right[-1]/scale, 'ok'


def score(left,right,scales,vector):
    size = np.linalg.norm(scales*vector)
    if size == 0:
        return None
    return float(np.dot(left@(vector/size),right@(vector/size)))


def projective_error(a,b,scale):
    a,b = a*scale,b*scale
    a,b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    distance=min(np.linalg.norm(a-b),np.linalg.norm(a+b))
    return float(2*np.arcsin(np.clip(distance/2.,0.,1.)))
