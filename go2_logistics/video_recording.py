"""Recoverable MP4 recording without changing the simulation clock.

The initialization metadata is written at the start, then completed fragments
are flushed every two seconds of video. An abruptly killed encoder may lose
its current fragment; this does not guarantee recovery after disk corruption.
"""
from pathlib import Path


def capture_frozen_frame(sim, capture, snapshot, *, tolerance=1e-7):
    """Block Kit's implicit stepping and verify native state around a capture.

    SimulationContext's step counter does not count physics driven directly
    by app.update(). The caller supplies independent backend state copies.
    """
    from .fleet_gui_check import check_frozen

    setting = '/app/player/playSimulations'
    previous = sim.get_setting(setting)
    if previous is None:
        raise RuntimeError('Cannot preserve unknown Kit physics stepping setting')
    frames = []
    sim.set_setting(setting, False)
    try:
        evidence = check_frozen(snapshot, lambda: frames.append(capture()),
                                pumps=1, tolerance=tolerance)
    finally:
        sim.set_setting(setting, previous)
    if frames[0] is None:
        raise RuntimeError('RGB recording returned no frame')
    return frames[0], evidence


def get_mission_writer(path, *, fps=5):
    import imageio.v2 as imageio

    return imageio.get_writer(str(Path(path)), fps=fps, codec='libx264', quality=8,
        macro_block_size=8, ffmpeg_params=[
            '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
            '-frag_duration', '2000000', '-flush_packets', '1'])
