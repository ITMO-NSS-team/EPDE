"""Use the validated innovations likelihood from the first optimizer step."""
from calibration_context import *


def estimate(y):
    original=F.A.profile_score
    def exact(cc,rhos,kappas,p1s,p2s,covariance_band):
        v,g,q=F.State.profile(cc,rhos,kappas,p1s,p2s);return v,g
    F.A.profile_score=exact
    try:model,info=Fit.estimate(y)
    finally:F.A.profile_score=original
    return model,dict(info,initial_likelihood_backend='native_stationary_innovations')
