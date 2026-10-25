"""Optional minimal projection of gait position targets, never measured state."""
import numpy as np


def project_leg_targets(target, limits, margin):
    target, limits = np.asarray(target, dtype=float), np.asarray(limits, dtype=float)
    if target.shape != (12,) or limits.shape != (12, 2):
        raise ValueError('Expected 12 leg targets and joint limits')
    if (not np.isfinite(target).all() or not np.isfinite(limits).all()
            or not np.isfinite(margin) or not 0. <= margin <= .20
            or np.any(limits[:, 1]-limits[:, 0] <= 2*margin)):
        raise ValueError('Invalid target guard bounds or margin')
    return np.clip(target, limits[:, 0]+margin, limits[:, 1]-margin)
