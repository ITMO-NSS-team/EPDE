"""Finite witnesses and exclusion certificates for a fixed weak support."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar


def forms(moments, covariances, indices, cutoff):
    # The last homogeneous coordinate multiplies the target with positive sign.
    signs = np.r_[-np.ones(len(indices) - 1), 1.]
    designs = [m[:, indices] * signs for m in moments]
    signal = np.stack([m.T @ m for m in designs])
    noise = np.stack([c[np.ix_(indices, indices)] * np.outer(signs, signs)
                      for c in covariances]) * cutoff
    return designs, signal, noise


def solve(moments, covariances, indices, cutoff):
    designs, signal, noise = forms(moments, covariances, indices, cutoff)
    matrices = signal - noise
    banks, dimension, _ = matrices.shape
    strength = signal + noise
    bank_scale = np.maximum(strength[:, -1, -1], np.finfo(float).tiny)
    scaled_strength = strength / bank_scale[:, None, None]
    diag = np.diag(scaled_strength.sum(axis=0))
    column_scale = 1 / np.sqrt(np.where(diag > 0, diag, 1.))
    normalized = matrices * np.outer(column_scale, column_scale) / bank_scale[:, None, None]
    normalized = (normalized + normalized.transpose(0, 2, 1)) / 2
    eye = np.eye(banks)

    def finite_witness(beta):
        vector = np.r_[beta, 1.]
        observed = np.array([np.linalg.norm(m @ vector)**2 for m in designs])
        allowance = np.einsum('i,bij,j->b', vector, noise, vector)
        # Check an interior margin relative to the original component scales.
        roundoff = 1e-11 * np.einsum('i,bij,j->b', abs(vector), abs(strength), abs(vector))
        if np.all(allowance - observed > roundoff):
            return dict(status='compatible', coefficients=beta.tolist(),
                        bank_ratios=(observed / allowance).tolist())
        return None

    # Prefer well-scaled finite least-squares coefficients to a far-away
    # feasible point. All least-squares fits use the observed fields only.
    weighted = [m / np.sqrt(s) for m, s in zip(designs, bank_scale)]
    for design in [np.vstack(weighted), *weighted]:
        beta = np.linalg.lstsq(design[:, :-1], -design[:, -1], rcond=None)[0]
        witness = finite_witness(beta)
        if witness:
            return witness
    def eig(weights):
        return np.linalg.eigvalsh(np.tensordot(weights, normalized, axes=1))[0]

    weights = [*eye, np.ones(banks) / banks]
    if banks >= 2:
        for i in range(banks):
            for j in range(i):
                def pair(t):
                    return t * eye[i] + (1-t) * eye[j]
                result = minimize_scalar(lambda t: -eig(pair(t)), bounds=(0., 1.),
                                         method='bounded', options={'xatol': 1e-13})
                weights.append(pair(result.x))
    if banks == 3:
        result = minimize(lambda w: -eig(w), weights[banks], method='SLSQP',
                          bounds=[(0., 1.)] * banks,
                          constraints=[dict(type='eq', fun=lambda w: w.sum()-1.)],
                          options={'ftol': 1e-13, 'maxiter': 150})
        w = np.clip(result.x, 0., 1.)
        if np.all(np.isfinite(w)) and w.sum() > 0:
            weights.append(w / w.sum())
    weights.sort(key=eig, reverse=True)
    for w in weights:
        mix = np.tensordot(w, normalized, axes=1)
        vals, vectors = np.linalg.eigh(mix)
        if vals[0] > 1e-11 * max(1., max(abs(vals))):
            return dict(status='excluded', weights=(w / bank_scale).tolist(),
                        scaled_min_eigenvalue=float(vals[0]))
        for col in vectors.T:
            direction = column_scale * col
            if abs(direction[-1]) > 1e-14 * np.linalg.norm(direction):
                witness = finite_witness(direction[:-1] / direction[-1])
                if witness:
                    return witness
            # A direction at infinity can provide a finite witness at large norm.
            for scale in (1., -1., 16., -16., 256., -256., 4096., -4096.):
                witness = finite_witness(direction[:-1] / column_scale[-1] * scale)
                if witness:
                    return witness
    return dict(status='unresolved')
