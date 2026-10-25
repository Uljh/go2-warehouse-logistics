"""Refine integration while preserving policy and render periods in seconds."""
import math


def refine_physics_timestep(cfg, factor):
    if factor not in (1,2):
        raise ValueError('Physics substeps factor must be 1 or 2')
    old_dt=cfg.sim.dt
    old_decimation=cfg.decimation
    if not math.isfinite(old_dt) or old_dt<=0 or old_decimation<=0:
        raise ValueError('Invalid original integration timing')
    cfg.sim.dt=old_dt/factor
    cfg.decimation=old_decimation*factor
    cfg.sim.render_interval*=factor
    # Sensor update periods are seconds and deliberately remain unchanged.
    return dict(factor=factor,physics_dt_s=cfg.sim.dt,decimation=cfg.decimation,
                control_period_s=cfg.sim.dt*cfg.decimation,
                render_period_s=cfg.sim.dt*cfg.sim.render_interval,
                original_physics_dt_s=old_dt)
