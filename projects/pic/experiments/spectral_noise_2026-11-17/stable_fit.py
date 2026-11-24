"""Same spectral estimator; cancellation-free AR precision near stationarity limits."""
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('isolated_spectral_fit',Path(__file__).with_name('noise_fit.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
np=base.np

def axis_precision(n,rho):
    k=np.arange(n);theta=np.pi*k/n
    numerator=(1-rho)**2+4*rho*np.sin(theta/2)**2 if rho>=0 else (1+rho)**2-4*rho*np.cos(theta/2)**2
    diagonal=numerator/((1-rho)*(1+rho))
    weights=np.sqrt(2./n)*np.ones(n);weights[0]=np.sqrt(1./n)
    endpoints=np.column_stack((weights*np.cos(theta/2),weights*np.cos(theta*(n-.5))))
    return diagonal,endpoints,rho/(1+rho)

base.axis_precision=axis_precision
Contrast=base.Contrast
estimate=base.estimate
