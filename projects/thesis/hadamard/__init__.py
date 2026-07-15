"""Data-driven Hadamard-stability diagnostics for the thesis.

Two halves, both estimated *purely from data* (no symbolic parsing of any
discovered equation, no ground-truth equation as reference, no forward
simulation of any model):

* ``forward_spectral``  -- forward well-posedness of the dynamics underlying
  the measured field, via DMD spectral abscissa + empirical dispersion
  ``Re s(k)`` and the Petrowsky criterion ``sup_k Re s(k) < inf``.
* ``inverse_stability`` -- continuous dependence of the discovery map
  ``data -> equation``: condition number, coefficient covariance, Monte-Carlo
  ensemble, and the perturbation-response Lipschitz slope.

See ``projects/thesis/hadamard_stability.md`` for the methodology.
"""
