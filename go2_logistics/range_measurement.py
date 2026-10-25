"""Seeded synthetic range imperfections; not a calibrated hardware model."""
from dataclasses import asdict, dataclass
import math
import random


@dataclass(frozen=True)
class RangeProfile:
    noise_sigma_m: float = .01
    dropout_probability: float = .03
    resolution_m: float = .005
    minimum_range_m: float = .05
    maximum_range_m: float = 5.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Non-finite range configuration')
        if not (self.noise_sigma_m >= 0 and 0 <= self.dropout_probability <= 1
                and self.resolution_m > 0 and 0 <= self.minimum_range_m < self.maximum_range_m):
            raise ValueError('Invalid range configuration')


class RangeMeasurementModel:
    def __init__(self, seed, profile=None):
        self.profile = profile or RangeProfile()
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.counts = dict(hit=0, no_return=0, dropout=0, out_of_range=0)

    def measure(self, raw_distance):
        cfg = self.profile
        if raw_distance is not None and (not math.isfinite(raw_distance) or raw_distance < 0):
            raise ValueError('Invalid raw range')
        distance = None
        if self.rng.random() < cfg.dropout_probability:
            status = 'dropout'
        elif raw_distance is None or raw_distance > cfg.maximum_range_m:
            status, distance = 'no_return', cfg.maximum_range_m
        elif raw_distance < cfg.minimum_range_m:
            status = 'out_of_range'
        else:
            noise = max(-3*cfg.noise_sigma_m, min(3*cfg.noise_sigma_m, self.rng.gauss(0, cfg.noise_sigma_m)))
            distance = round((raw_distance+noise)/cfg.resolution_m)*cfg.resolution_m
            status = 'hit'
            if not cfg.minimum_range_m <= distance <= cfg.maximum_range_m:
                status, distance = 'out_of_range', None
        self.counts[status] += 1
        return dict(status=status, distance_m=distance)

    def metadata(self):
        return dict(model='synthetic_bounded_gaussian_quantization_independent_dropout',
                    seed=self.seed, configuration=asdict(self.profile), counts=dict(self.counts),
                    calibrated_to_hardware=False, invalid_ray_clears_map=False)


def ray_observation(start, direction, layer, measurement):
    """Invalid beams contribute neither occupied endpoints nor free space."""
    if measurement['status'] in ('dropout', 'out_of_range'):
        return None, None
    distance = measurement['distance_m']
    point = (start[0]+direction[0]*distance, start[1]+direction[1]*distance)
    hit = (*point, layer, distance) if measurement['status'] == 'hit' else None
    return hit, (tuple(start), point, layer)
