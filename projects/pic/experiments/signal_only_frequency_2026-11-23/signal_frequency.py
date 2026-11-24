"""Two candidate angular frequencies from a scalar uniformly sampled signal."""
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from epde.interface.token_frequencies import _merge_peaks, _spectrum_peaks


def recurrence_frequency(values, spacing, anchor):
    period = 2*np.pi/(spacing*anchor)
    x = values[len(values)//2:] if len(values)>=4*period else values
    sigma = max(.5, period/32)
    trim = int(np.ceil(2*sigma))
    x = gaussian_filter1d(x, sigma)[trim:len(x)-trim]
    n = len(x)
    low = max(2,int(np.ceil(.75*period)))
    high = min(2*n//3,int(np.floor(1.25*period)))
    if high-low<2:
        return None
    x = x-np.mean(x)
    lags = np.arange(low,high+1)
    sizes = n-lags
    sums = np.r_[0.,np.cumsum(x)]
    squares = np.r_[0.,np.cumsum(x*x)]
    left, right = sums[sizes], sums[n]-sums[lags]
    covariance = correlate(x,x,mode='full',method='fft')[n-1+lags]-left*right/sizes
    var1 = squares[sizes]-left*left/sizes
    var2 = squares[n]-squares[lags]-right*right/sizes
    scale = np.sqrt(np.maximum(var1,0.)*np.maximum(var2,0.))
    correlations = np.divide(covariance,scale,out=np.full(len(lags),-np.inf),where=scale>0)
    j = int(np.argmax(correlations))
    if j==0 or j==len(lags)-1 or correlations[j]<.8:
        return None
    before,middle,after = correlations[j-1:j+2]
    curvature = before-2*middle+after
    if not np.isfinite(curvature) or curvature>=0:
        return None
    offset = np.clip(.5*(before-after)/curvature,-.5,.5)
    return float(2*np.pi/(spacing*(lags[j]+offset)))


def propose(values, spacing):
    """Return the native list and repeat-period alternative, strongest peak first."""
    x = np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<4 or not np.isfinite(x).all() or not np.isfinite(spacing) or spacing<=0:
        raise ValueError('Supply one finite, uniformly sampled series and positive sample spacing.')
    peaks = _spectrum_peaks(x[None,:],spacing)
    clusters = sorted(_merge_peaks(peaks or []),key=lambda c:(-c[0],c[1]))[:2]
    native = tuple(float(2*np.pi*f) for energy,f in clusters)
    if not native or (len(clusters)==2 and clusters[1][0]>=.1*clusters[0][0]):
        return native,native,False
    alternative = recurrence_frequency(x,spacing,native[0])
    if alternative is None or abs(alternative-native[0])<.005*native[0]:
        return native,native,False
    return native,(native[0],alternative),True
