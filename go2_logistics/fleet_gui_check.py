"""Native GUI clock checks using backend physics state, not cached asset data."""
import numpy as np


def check_frozen(snapshot, pump, *, pumps=12, tolerance=1e-7):
    before = snapshot()
    for _ in range(pumps):
        pump()
    after = snapshot()
    if before.keys() != after.keys():
        raise RuntimeError('Physics snapshot schema changed')
    errors = {}
    for key in before:
        a, b = np.asarray(before[key]), np.asarray(after[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f'Invalid physics snapshot: {key}')
        errors[key] = float(np.max(np.abs(a-b))) if a.size else 0.
    if any(error > tolerance for error in errors.values()):
        raise RuntimeError(f'GUI advanced frozen physics: {errors}')
    return dict(pumps=pumps, tolerance=tolerance, max_state_errors=errors, passed=True)


def backend_snapshot(sim, robots, objects):
    import warp as wp
    result = {'physics_step_count': np.asarray([sim.get_physics_step_count()])}
    for name, robot in robots.items():
        for method in ('get_root_transforms', 'get_root_velocities', 'get_dof_positions', 'get_dof_velocities'):
            result[f'{name}/{method}'] = wp.to_torch(getattr(robot.root_view, method)()).detach().cpu().numpy().copy()
    for name, obj in objects.items():
        for method in ('get_transforms', 'get_velocities'):
            result[f'{name}/{method}'] = wp.to_torch(getattr(obj.root_view, method)()).detach().cpu().numpy().copy()
    return result
