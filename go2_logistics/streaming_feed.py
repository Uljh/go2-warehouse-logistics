"""Finite, physically pre-created conveyor feed for long-running warehouse runs.

The feed deliberately has no runtime spawning, teleporting, or object reset.  A
run starts with a finite queue of rigid parcels already authored on the belt;
the belt and receiving gate move them by contact and friction.  This module is
the identity/termination ledger used by the runtime around that physical feed.

The ledger is intentionally separate from the dispatcher.  A camera detection
must be recorded before a code can be claimed, and a code can only be delivered
once.  Array position is useful for configuring the USD prim path, but it is
never sufficient evidence for an order claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .sorting_layout import ParcelCode


MIN_STREAM_CODES = 6
MAX_STREAM_CODES = 10
DEFAULT_STREAM_SPACING_M = 0.32
PARCEL_LENGTH_M = 0.10
PARCEL_WIDTH_M = 0.035
PARCEL_HEIGHT_M = 0.08


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


@dataclass(frozen=True)
class FeedParcel:
    """The physical manifest for one pre-created parcel."""

    code: str
    index: int
    prim_path: str
    queue_x_m: float
    queue_y_m: float
    queue_z_m: float


@dataclass(frozen=True)
class StreamingFeedPlan:
    """Geometry and identity manifest for a bounded continuous-feed trial.

    ``codes`` may contain six to ten parcels so a shorter 6--8 parcel physical
    smoke run is possible.  The intended production recipe is eight to ten.
    All rigid bodies are created during scene construction.  ``queue_length``
    is therefore a real belt length, not an instruction to create objects at
    later ticks.
    """

    codes: tuple[str, ...]
    spacing_m: float = DEFAULT_STREAM_SPACING_M
    belt_lead_m: float = 0.60
    parcel_center_offset_m: float = 0.20
    _parcels: tuple[FeedParcel, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        codes = tuple(str(code).strip() for code in self.codes)
        if not MIN_STREAM_CODES <= len(codes) <= MAX_STREAM_CODES:
            raise ValueError(
                f"streaming feed requires {MIN_STREAM_CODES}..{MAX_STREAM_CODES} unique parcels"
            )
        if len(set(codes)) != len(codes):
            raise ValueError("streaming feed parcel identities must be unique")
        for code in codes:
            ParcelCode.parse(code)
        spacing = _finite_positive(self.spacing_m, "spacing_m")
        lead = _finite_positive(self.belt_lead_m, "belt_lead_m")
        offset = _finite_positive(self.parcel_center_offset_m, "parcel_center_offset_m")
        if spacing < PARCEL_LENGTH_M + 0.02:
            raise ValueError("spacing_m leaves no physical clearance between parcels")
        if offset <= PARCEL_LENGTH_M / 2:
            raise ValueError("parcel_center_offset_m must leave a receiving lead")
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "belt_lead_m", lead)
        object.__setattr__(self, "parcel_center_offset_m", offset)
        object.__setattr__(
            self,
            "_parcels",
            tuple(
                FeedParcel(
                    code=code,
                    index=index,
                    prim_path="/World/envs/env_0/Parcel" + (str(index) if index else ""),
                    queue_x_m=offset + index * spacing,
                    queue_y_m=0.0,
                    queue_z_m=PARCEL_HEIGHT_M / 2,
                )
                for index, code in enumerate(codes)
            ),
        )

    @property
    def production_recipe(self) -> bool:
        """Whether this recipe meets the requested 8--10 parcel target."""

        return len(self.codes) >= 8

    @property
    def queue_length_m(self) -> float:
        """Belt length required for the pre-created queue and a lead margin."""

        last_center = self.parcel_center_offset_m + (len(self.codes) - 1) * self.spacing_m
        return self.belt_lead_m + last_center + PARCEL_LENGTH_M / 2

    @property
    def parcels(self) -> tuple[FeedParcel, ...]:
        return self._parcels

    def physical_manifest(self, pickup_x: float, pickup_y: float, belt_top_m: float = 0.025) -> tuple[FeedParcel, ...]:
        """Return world-space positions for scene authoring.

        This function only supplies initial rigid-body poses.  It never writes
        a pose during a mission and never marks an item as observed or claimed.
        """

        pickup_x = float(pickup_x)
        pickup_y = float(pickup_y)
        belt_top_m = float(belt_top_m)
        if not all(math.isfinite(v) for v in (pickup_x, pickup_y, belt_top_m)):
            raise ValueError("pickup geometry must be finite")
        return tuple(
            FeedParcel(
                code=item.code,
                index=item.index,
                prim_path=item.prim_path,
                queue_x_m=pickup_x + item.queue_x_m,
                queue_y_m=pickup_y + item.queue_y_m,
                queue_z_m=belt_top_m + PARCEL_HEIGHT_M / 2,
            )
            for item in self.parcels
        )


class FeedLedgerError(RuntimeError):
    """A feed event violates the physical identity/termination contract."""


class StreamingFeedAudit:
    """Event ledger for a finite physical feed.

    The runtime should call ``observe_receiving`` from the decoded RGB-D
    detection, then ``claim_visual`` only after the dispatcher atomically owns
    that same code.  ``mark_delivered`` is called after placement verification.
    Position samples are optional audit data and do not change state.
    """

    def __init__(self, codes: Iterable[str], *, run_id: str = "",
                 receiving_xy: Sequence[float] | None = None,
                 receiving_radius_m: float = .08) -> None:
        plan = StreamingFeedPlan(tuple(codes))
        self.codes = plan.codes
        self.run_id = str(run_id)
        if receiving_xy is not None:
            xy = tuple(float(value) for value in receiving_xy)
            if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
                raise ValueError('receiving_xy must contain two finite coordinates')
            self.receiving_xy = xy
            self.receiving_radius_m = _finite_positive(receiving_radius_m, 'receiving_radius_m')
        else:
            self.receiving_xy = None
            self.receiving_radius_m = None
        self.state = {code: "source" for code in self.codes}
        self.events: list[dict] = []
        self.position_samples: dict[str, int] = {code: 0 for code in self.codes}
        self.receiving_observation_counts: dict[str, int] = {code: 0 for code in self.codes}
        self.claim_counts: dict[str, int] = {code: 0 for code in self.codes}
        self.delivery_counts: dict[str, int] = {code: 0 for code in self.codes}
        self.duplicate_events: list[dict] = []
        self.preloaded: set[str] = set()
        self.position_bounds: dict[str, dict] = {
            code: {"samples": 0, "first": None, "last": None,
                   "min": None, "max": None, "max_step_m": 0.0}
            for code in self.codes
        }

    def _code(self, code: str) -> str:
        code = str(code).strip()
        if code not in self.state:
            raise FeedLedgerError(f"unknown visual parcel identity: {code}")
        return code

    @staticmethod
    def _validate_tick(tick: int) -> None:
        if not isinstance(tick, int) or tick < 0:
            raise FeedLedgerError("feed event tick must be a nonnegative integer")

    def _record(self, event: str, code: str | None, tick: int | None, **extra) -> dict:
        if tick is not None:
            self._validate_tick(tick)
        row = {"event": event, "code": code, "tick": tick, **extra}
        self.events.append(row)
        return row

    def mark_on_belt(self, code: str, *, tick: int) -> None:
        self._validate_tick(tick)
        code = self._code(code)
        if code in self.preloaded:
            return
        if self.state[code] != "source":
            if self.state[code] in {"receiving", "claimed", "carrying", "delivered"}:
                return
            raise FeedLedgerError(f"cannot release {code} from state {self.state[code]}")
        # All parcels are physically pre-created on the belt at scene startup;
        # this is an audit event only.  A parcel remains in source state until
        # the camera decodes it at the receiving station.
        self.preloaded.add(code)
        self._record("physical_queue_preloaded", code, tick)

    def observe_receiving(self, detection: Mapping, *, tick: int) -> bool:
        """Record a decoded receiving observation; return whether it is new."""

        self._validate_tick(tick)
        if (not isinstance(detection, Mapping)
                or detection.get('source') != 'rendered_rgb_depth'):
            raise FeedLedgerError('receiving identity requires a rendered RGB-D detection')
        code = self._code(detection.get('parcel_code', ''))
        center = detection.get('center_world')
        if (not isinstance(center, (tuple, list)) or len(center) != 3
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in center)):
            raise FeedLedgerError('receiving identity requires a finite RGB-D center')
        if self.receiving_xy is not None and math.dist(center[:2], self.receiving_xy) > self.receiving_radius_m:
            raise FeedLedgerError('rendered parcel center is outside the receiving zone')
        current = self.state[code]
        if current in {"delivered", "failed", "claimed", "carrying"}:
            return False
        if code not in self.preloaded:
            raise FeedLedgerError(f"parcel {code} was visually observed before physical release")
        first = current != "receiving"
        self.receiving_observation_counts[code] += 1
        if not first:
            return False
        self.state[code] = "receiving"
        self._record("visual_receiving_observation", code, tick,
                     source=detection['source'], center_world=list(center))
        return first

    def claim_visual(self, code: str, *, tick: int) -> None:
        """Accept a dispatcher claim only for the currently observed code."""

        self._validate_tick(tick)
        code = self._code(code)
        if self.state[code] != "receiving":
            if self.state[code] in {"claimed", "carrying", "delivered"}:
                event = self._record("duplicate_claim_rejected", code, tick, state=self.state[code])
                self.duplicate_events.append(event)
            raise FeedLedgerError(f"visual claim for {code} requires receiving observation")
        self.state[code] = "claimed"
        self.claim_counts[code] += 1
        self._record("claimed_after_visual_observation", code, tick)

    def mark_carrying(self, code: str, *, tick: int) -> None:
        self._validate_tick(tick)
        code = self._code(code)
        if self.state[code] != "claimed":
            raise FeedLedgerError(f"cannot mark {code} carrying from {self.state[code]}")
        self.state[code] = "carrying"
        self._record("carrying", code, tick)

    def mark_delivered(self, code: str, *, tick: int) -> None:
        self._validate_tick(tick)
        code = self._code(code)
        if self.state[code] == "delivered":
            event = self._record("duplicate_delivery_rejected", code, tick, state="delivered")
            self.duplicate_events.append(event)
            raise FeedLedgerError(f"duplicate delivery for {code}")
        if self.state[code] != "carrying":
            raise FeedLedgerError(f"cannot deliver {code} from {self.state[code]}")
        self.state[code] = "delivered"
        self.delivery_counts[code] += 1
        self._record("delivered_after_verified_placement", code, tick)

    def mark_failed(self, code: str, *, tick: int, reason: str) -> None:
        self._validate_tick(tick)
        code = self._code(code)
        if self.state[code] in {"delivered", "failed"}:
            raise FeedLedgerError(f"cannot fail terminal parcel {code}")
        self.state[code] = "failed"
        self._record("failed", code, tick, reason=str(reason))

    def record_position(self, code: str, position: Sequence[float], *, tick: int) -> None:
        self._validate_tick(tick)
        code = self._code(code)
        values = tuple(float(value) for value in position)
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise FeedLedgerError(f"invalid physical position sample for {code}")
        self.position_samples[code] += 1
        summary = self.position_bounds[code]
        if summary["first"] is None:
            summary["first"] = list(values)
            summary["first_tick"] = tick
            summary["min"] = list(values)
            summary["max"] = list(values)
        else:
            previous = summary["last"]
            summary["max_step_m"] = max(summary["max_step_m"], math.dist(previous, values))
            summary["min"] = [min(a, b) for a, b in zip(summary["min"], values)]
            summary["max"] = [max(a, b) for a, b in zip(summary["max"], values)]
        summary["last"] = list(values)
        summary["last_tick"] = tick
        summary["samples"] += 1

    @property
    def source_exhausted(self) -> bool:
        return all(value != "source" for value in self.state.values())

    @property
    def all_delivered(self) -> bool:
        return bool(self.state) and all(value == "delivered" for value in self.state.values())

    @property
    def complete(self) -> bool:
        return self.source_exhausted and self.all_delivered

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "codes": list(self.codes),
            "state": dict(self.state),
            "preloaded_codes": sorted(self.preloaded),
            "source_exhausted": self.source_exhausted,
            "all_delivered": self.all_delivered,
            "complete": self.complete,
            "claim_counts": dict(self.claim_counts),
            "receiving_observation_counts": dict(self.receiving_observation_counts),
            "delivery_counts": dict(self.delivery_counts),
            "position_sample_counts": dict(self.position_samples),
            "position_bounds": {code: dict(summary) for code, summary in self.position_bounds.items()},
            "receiving_zone_xy_m": list(self.receiving_xy) if self.receiving_xy is not None else None,
            "receiving_zone_radius_m": self.receiving_radius_m,
            "duplicate_events": list(self.duplicate_events),
            "events": list(self.events),
        }

    def write_json(self, path: str | Path) -> Path:
        """Persist an auditable, compact feed summary after or during a run."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return path

