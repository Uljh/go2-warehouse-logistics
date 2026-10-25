"""Continuous recovery commands when an inherited policy target is out of range.

The physical joint bounds are unchanged. A target already outside the position
box cannot both enter the box immediately and respect an arbitrary slew bound.
This candidate projects the destination, then approaches it at the original
rate. Any inherited violation decreases without creating a new violation.
"""
import math


def continuous_recovery_target(previous, desired, lower, upper, max_step):
    import torch
    if (previous.ndim != 1 or previous.shape != desired.shape
            or previous.shape != lower.shape or previous.shape != upper.shape):
        raise ValueError('Recovery target vectors must have the same one-dimensional shape')
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError('Recovery target slew step must be finite and positive')
    if (not all(bool(torch.isfinite(value).all()) for value in (previous,desired,lower,upper))
            or not bool((lower < upper).all())):
        raise ValueError('Recovery targets and bounds must be finite and ordered')
    destination = torch.maximum(torch.minimum(desired, upper), lower)
    return previous + (destination-previous).clamp(-max_step,max_step)


def bound_violation(target, lower, upper):
    return ((lower-target).clamp(min=0)+(target-upper).clamp(min=0)).max()
