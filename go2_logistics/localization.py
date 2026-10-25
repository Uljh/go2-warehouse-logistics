"""Time-correlated synthetic localization error, not odometry or SLAM.

The simulator supplies the reference pose. A bounded Ornstein-Uhlenbeck error
is added once per control tick. All consumers share that immutable observation;
range endpoints are reconstructed in its frame instead of using truth hits.
"""
from dataclasses import asdict, dataclass
import math
import random
from .interfaces import Pose2D


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class LocalizationProfile:
    position_sigma_m: float = .005
    yaw_sigma_rad: float = math.radians(.1)
    correlation_time_s: float = 10.
    maximum_position_error_m: float = .02
    maximum_yaw_error_rad: float = math.radians(.5)

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('Localization configuration must be positive and finite')


@dataclass(frozen=True)
class PoseObservation:
    tick: int
    sim_time_s: float
    truth: Pose2D
    estimate: Pose2D

    def ray_in_map(self, world_direction):
        """Physical world beam -> body-relative beam -> estimated map beam."""
        angle = wrap(self.estimate.yaw - self.truth.yaw)
        c, s = math.cos(angle), math.sin(angle)
        dx, dy = world_direction[:2]
        return ((self.estimate.x, self.estimate.y), (c*dx-s*dy, s*dx+c*dy, 0.))

    def record(self):
        return dict(tick=self.tick, sim_time_s=self.sim_time_s,
                    truth_pose=asdict(self.truth), estimated_pose=asdict(self.estimate))


class LocalizationModel:
    def __init__(self, seed, profile=None):
        self.seed = int(seed)
        self.profile = profile or LocalizationProfile()
        self.rng = random.Random(self.seed)
        self.error = [0., 0., 0.]
        self.observation = None
        self.samples = 0

    def update(self, tick, sim_time_s, truth):
        if (not isinstance(tick, int) or tick < 0 or not math.isfinite(sim_time_s)
                or sim_time_s < 0 or not all(math.isfinite(v) for v in asdict(truth).values())):
            raise ValueError('Invalid localization input')
        previous = self.observation
        if previous is not None:
            if tick == previous.tick:
                if sim_time_s != previous.sim_time_s or truth != previous.truth:
                    raise ValueError('Pose changed without advancing the localization tick')
                return previous
            if tick != previous.tick + 1 or sim_time_s <= previous.sim_time_s:
                raise ValueError('Localization must advance on every consecutive control tick')
        cfg = self.profile
        decay = math.exp(-(sim_time_s-previous.sim_time_s)/cfg.correlation_time_s) if previous else 0.
        scale = math.sqrt(max(0., 1-decay*decay))
        sigmas = (cfg.position_sigma_m, cfg.position_sigma_m, cfg.yaw_sigma_rad)
        self.error = [decay*e + scale*self.rng.gauss(0., sigma) for e, sigma in zip(self.error, sigmas)]
        radius = math.hypot(*self.error[:2])
        if radius > cfg.maximum_position_error_m:
            self.error[:2] = [e*cfg.maximum_position_error_m/radius for e in self.error[:2]]
        self.error[2] = max(-cfg.maximum_yaw_error_rad, min(cfg.maximum_yaw_error_rad, self.error[2]))
        estimate = Pose2D(truth.x+self.error[0], truth.y+self.error[1], wrap(truth.yaw+self.error[2]))
        self.observation = PoseObservation(tick, sim_time_s, truth, estimate)
        self.samples += 1
        return self.observation

    def metadata(self):
        return dict(model='synthetic_bounded_time_correlated_pose_error', seed=self.seed,
                    configuration=asdict(self.profile), samples=self.samples,
                    calibrated_to_hardware=False, source='simulator_ground_truth_plus_synthetic_error',
                    update='once per consecutive control tick; cached for all navigation and scan consumers',
                    scope='fleet bridge planar navigation/docking and range projection; not gait or arm state estimation')
