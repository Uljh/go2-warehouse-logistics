"""Run the trained Isaac Lab Go2 policy inside the logistics mission.

This module deliberately keeps Isaac imports inside :func:`main`.  The small
``Go2PolicyBridge`` class can therefore be imported by the dependency-free
tests, while an Isaac Lab installation supplies the real environment and
checkpoint at runtime.

The official Go2 policy is a velocity-tracking locomotion policy.  It does not
accept a Cartesian goal directly, so the bridge converts the current waypoint
error into a bounded ``base_velocity`` command before every policy inference.
That gives us a useful navigation baseline without retraining the policy.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .interfaces import Pose2D
    from .isaac_entrypoint import build_mission
except ImportError:  # Script execution through ``isaaclab.bat -p``.
    from go2_logistics.interfaces import Pose2D
    from go2_logistics.isaac_entrypoint import build_mission


@dataclass(frozen=True)
class BridgeConfig:
    """Control limits for the waypoint-to-velocity adapter."""

    max_control_steps: int = 240
    position_tolerance: float = 0.35
    yaw_tolerance: float = 0.35
    linear_gain: float = 0.8
    yaw_gain: float = 1.2
    max_linear_speed: float = 0.8
    max_lateral_speed: float = 0.5
    max_yaw_speed: float = 1.0
    minimum_lateral_command: float = 0.22
    minimum_longitudinal_command: float = 0.22
    minimum_docking_lateral_command: float | None = None


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _to_float(value: Any) -> float:
    """Convert a scalar tensor/array value without importing torch or numpy."""
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _tensor(value: Any) -> Any:
    """Support Isaac Lab tensor wrappers and plain torch tensors."""
    return getattr(value, "torch", value)


class Go2PolicyBridge:
    """Expose an Isaac Lab velocity policy through the mission robot interface.

    ``env`` is the ``RslRlVecEnvWrapper`` used by Isaac Lab's official play
    script.  The implementation intentionally uses the public scene and
    command-manager objects instead of private simulator internals.
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        config: BridgeConfig | None = None,
        post_step: Any = None,
        pre_step: Any = None,
    ):
        self.env = env
        self.policy = policy
        self.config = config or BridgeConfig()
        self.post_step = post_step
        self.pre_step = pre_step
        self._last_obs: Any = None
        self._last_dones: Any = None

    def pose(self) -> Pose2D:
        """Read env 0's base pose in the local warehouse frame."""
        base = self.env.unwrapped.scene["robot"]
        pos = _tensor(base.data.root_pos_w)[0]
        origins = _tensor(self.env.unwrapped.scene.env_origins)[0]
        quat = _tensor(base.data.root_quat_w)[0]
        # Isaac Lab 3.x exposes world quaternions as (x, y, z, w).
        x, y, z, w = (_to_float(quat[i]) for i in range(4))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return Pose2D(_to_float(pos[0] - origins[0]), _to_float(pos[1] - origins[1]), yaw)

    def _set_velocity_command(self, goal: Pose2D) -> None:
        """Set a body-frame velocity command for env 0.

        The command manager normally resamples this term periodically.  Writing
        the command immediately before observation computation overrides that
        random target for the current navigation segment, while the policy still
        receives the same observation layout it was trained with.
        """
        current = self.pose()
        dx = goal.x - current.x
        dy = goal.y - current.y
        cos_yaw = math.cos(current.yaw)
        sin_yaw = math.sin(current.yaw)
        # World error -> body frame.
        ex = cos_yaw * dx + sin_yaw * dy
        ey = -sin_yaw * dx + cos_yaw * dy
        command = self.env.unwrapped.command_manager.get_term("base_velocity").command
        command[0, 0] = max(-self.config.max_linear_speed, min(self.config.max_linear_speed, self.config.linear_gain * ex))
        command[0, 1] = max(-self.config.max_lateral_speed, min(self.config.max_lateral_speed, self.config.linear_gain * ey))
        # The published gait has a small-command standing region. Keep a
        # resolvable velocity inside the radial docking tolerance. A fixed
        # 4cm per-axis deadband can stop diagonally outside a 4cm goal radius.
        # Fine docking corrections are local. Applying that small deadband on
        # long aisle legs amplifies tiny lateral errors into gait commands.
        near_goal=math.hypot(dx,dy)<max(.12,2*self.config.position_tolerance)
        deadband=min(.04,self.config.position_tolerance*.5) if near_goal else .04
        for axis, error in ((0, ex), (1, ey)):
            minimum = self.config.minimum_longitudinal_command if axis == 0 else self.config.minimum_lateral_command
            if axis == 1 and near_goal and self.config.minimum_docking_lateral_command is not None:
                minimum = self.config.minimum_docking_lateral_command
            if abs(error) > deadband and abs(float(command[0, axis])) < minimum:
                command[0, axis] = math.copysign(minimum, error)
        yaw_error = _wrap_angle(goal.yaw - current.yaw)
        command[0, 2] = max(-self.config.max_yaw_speed, min(self.config.max_yaw_speed, self.config.yaw_gain * yaw_error))

    def _zero_command(self) -> None:
        command = self.env.unwrapped.command_manager.get_term("base_velocity").command
        command[0, :] = 0.0

    def _policy_step(self) -> None:
        if callable(self.pre_step):
            self.pre_step()
        # Recompute after writing the command so velocity_commands reflects the
        # waypoint rather than the command randomly sampled by Isaac Lab.
        self._last_obs = self.env.get_observations()
        with _inference_mode():
            actions = self.policy(self._last_obs)
        self._last_obs, _, dones, _ = self.env.step(actions)
        self._last_dones = dones
        if bool(_to_float(dones[0])):
            print(f"POLICY_TERMINATION {self.env.unwrapped.termination_manager}", flush=True)
            print(f"POLICY_TERMINATION terms={self.env.unwrapped.termination_manager._term_dones}", flush=True)
            raise RuntimeError("Robot episode terminated during mission (fall or timeout); refusing silent reset")
        if _to_float(_tensor(self.env.unwrapped.scene['robot'].data.projected_gravity_b)[0, 2]) > -.85:
            raise RuntimeError("Robot tilted beyond 32 degrees; refusing to continue manipulation after instability")
        if callable(self.post_step):
            self.post_step(self.pose())
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(dones)

    def navigate(self, goal: Pose2D) -> bool:
        """Drive to ``goal`` with the trained policy and return success."""
        start = self.pose()
        print(
            f"POLICY_NAVIGATION start=({start.x:.2f},{start.y:.2f}) goal=({goal.x:.2f},{goal.y:.2f})",
            flush=True,
        )
        for _ in range(self.config.max_control_steps):
            self._set_velocity_command(goal)
            current = self.pose()
            if os.environ.get("GO2_DEBUG") and (_ == 0 or (_ + 1) % 20 == 0):
                print(
                    "GO2_DEBUG step={} pose=({:.3f},{:.3f},{:.3f}) goal=({:.3f},{:.3f},{:.3f})".format(
                        _ + 1, current.x, current.y, current.yaw, goal.x, goal.y, goal.yaw
                    ),
                    flush=True,
                )
            distance = math.hypot(goal.x - current.x, goal.y - current.y)
            yaw_error = abs(_wrap_angle(goal.yaw - current.yaw))
            if distance <= self.config.position_tolerance and yaw_error <= self.config.yaw_tolerance:
                self._zero_command()
                return True
            self._policy_step()
            if (_ + 1) % 60 == 0:
                current = self.pose()
                print(
                    f"POLICY_PROGRESS step={_ + 1} pose=({current.x:.2f},{current.y:.2f}) "
                    f"distance={math.hypot(goal.x-current.x, goal.y-current.y):.2f} "
                    f"height={_to_float(_tensor(self.env.unwrapped.scene['robot'].data.root_pos_w)[0, 2]):.3f} "
                    f"gravity={_tensor(self.env.unwrapped.scene['robot'].data.projected_gravity_b)[0].tolist()}",
                    flush=True,
                )
        self._zero_command()
        current = self.pose()
        print(
            f"POLICY_NAVIGATION timeout pose=({current.x:.2f},{current.y:.2f}) "
            f"distance={math.hypot(goal.x-current.x, goal.y-current.y):.2f}",
            flush=True,
        )
        return False


class _inference_mode:
    """Tiny optional context manager to keep this module importable without torch."""

    def __enter__(self):
        try:
            import torch
        except ImportError:
            self._ctx = None
        else:
            self._ctx = torch.inference_mode()
            self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False


class FixedSceneArm:
    """Minimal arm boundary for the first Isaac demo.

    This is intentionally explicit: it models the task transition while the
    Go2 locomotion policy is validated.  Replace it with cuRobo/MoveIt once a
    robot asset with an arm is added to the scene.
    """

    def __init__(self):
        self.held = False

    def grasp(self, object_id: str) -> bool:
        del object_id
        self.held = True
        return True

    def release(self) -> bool:
        was_held = self.held
        self.held = False
        return was_held


class VisualPayloadAdapter:
    """Animate the authored task box during a live policy rollout.

    This adapter is intentionally a visual layer: the Go2 articulation remains
    the only policy controlled body, while the box follows the measured base
    pose after the mission's grasp transition.  Keeping this boundary explicit
    makes it straightforward to replace with a contact accurate cuRobo/Isaac
    Manipulator controller later.
    """

    _WAITING_COLOR = (0.92, 0.08, 0.035)
    _CARRYING_COLOR = (1.0, 0.78, 0.08)
    _RELEASED_COLOR = (0.10, 0.82, 0.38)

    def __init__(self, stage: Any, pickup: Pose2D, dropoff: Pose2D, gripper: Any = None):
        from pxr import Gf, UsdGeom

        self.stage = stage
        self.Gf = Gf
        self.UsdGeom = UsdGeom
        candidates = (
            "/World/Warehouse/Task/PickupBox",
            "/World/Warehouse/World/Task/PickupBox",
            "/World/Task/PickupBox",
        )
        self.path = next((path for path in candidates if stage.GetPrimAtPath(path).IsValid()), None)
        if self.path is None:
            # A proxy keeps visual playback useful even when a scene was
            # authored without the task anchors.  It follows the same state
            # machine as the authored pickup box.
            self.path = "/World/Task/VisualPayload"
            proxy = UsdGeom.Cube.Define(stage, self.path)
            proxy.CreateSizeAttr(1.0)
            proxy.AddScaleOp().Set(Gf.Vec3f(0.62, 0.62, 0.56))
            self.box = proxy.GetPrim()
            print(f"POLICY_VISUAL payload=created path={self.path}", flush=True)
        else:
            self.box = stage.GetPrimAtPath(self.path)
            print(f"POLICY_VISUAL payload={self.path}", flush=True)
        self.xform = UsdGeom.XformCommonAPI(self.box)
        self.pickup = pickup
        self.dropoff = dropoff
        self.gripper = gripper
        self.held = False
        self.state = "waiting"
        self.last_pose = pickup
        self._set_state("waiting")
        self._set_color(self._WAITING_COLOR)

    def _set_state(self, state: str) -> None:
        self.state = state
        self.box.SetCustomDataByKey("payload_state", state)

    def _set_color(self, color: tuple[float, float, float]) -> None:
        if self.box.IsA(self.UsdGeom.Gprim):
            self.UsdGeom.Gprim(self.box).CreateDisplayColorAttr().Set([self.Gf.Vec3f(*color)])

    def _set_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        self.xform.SetTranslate(self.Gf.Vec3d(x, y, z))
        self.xform.SetRotate((0.0, 0.0, math.degrees(yaw)))

    def prepare(self) -> None:
        self.held = False
        if self.gripper is not None:
            self.gripper.reset()
        self._set_pose(self.pickup.x, self.pickup.y, 0.30, self.pickup.yaw)
        self._set_state("waiting")
        self._set_color(self._WAITING_COLOR)

    def _carry_pose(self, pose: Pose2D) -> tuple[float, float, float, float]:
        # Offset the carton toward the deck/front of the base and rotate it
        # with the robot, so it reads as a carried payload rather than a box
        # teleporting at the robot origin.
        forward = 0.34
        x = pose.x + math.cos(pose.yaw) * forward
        y = pose.y + math.sin(pose.yaw) * forward
        return x, y, 1.22, pose.yaw

    def sync(self, pose: Pose2D) -> None:
        self.last_pose = pose
        if self.held:
            self._set_pose(*self._carry_pose(pose))

    def grasp(self, object_id: str) -> bool:
        del object_id
        self.held = True
        if self.gripper is not None:
            self.gripper.set_state("grasp")
        self.sync(self.last_pose)
        self._set_state("carried")
        self._set_color(self._CARRYING_COLOR)
        print(
            f"POLICY_GRASP visual_payload_attached=true path={self.path} "
            f"pose=({self.last_pose.x:.2f},{self.last_pose.y:.2f})",
            flush=True,
        )
        return True

    def release(self) -> bool:
        if not self.held:
            return False
        self.held = False
        if self.gripper is not None:
            self.gripper.set_state("released")
        self._set_pose(self.dropoff.x, self.dropoff.y, 0.30, self.dropoff.yaw)
        self._set_state("released")
        self._set_color(self._RELEASED_COLOR)
        print(
            f"POLICY_RELEASE visual_payload_placed=true path={self.path} "
            f"pose=({self.dropoff.x:.2f},{self.dropoff.y:.2f})",
            flush=True,
        )
        return True


class PolicyMissionHud:
    """Compact in-viewport readout for the policy-backed mission."""

    def __init__(self):
        import omni.ui as ui

        self.window = ui.Window("GO2 POLICY / LOGISTICS", width=340, height=240)
        self.window.setPosition(20, 110)
        self._run_callback = None
        self.follow_camera = True
        with self.window.frame:
            with ui.VStack(spacing=5, height=0):
                self.state = ui.Label("MISSION  starting", style={"font_size": 18, "color": 0xFFFFFFFF})
                self.target = ui.Label("TARGET  --", style={"font_size": 14, "color": 0xFFB7D7FF})
                self.pose = ui.Label("GO2  (0.00, 0.00, 0.00)", style={"font_size": 14, "color": 0xFFE0E0E0})
                self.payload = ui.Label("PAYLOAD  waiting", style={"font_size": 14, "color": 0xFFFFC080})
                self.recovery = ui.Label("RECOVERY  0", style={"font_size": 14, "color": 0xFF80E0A0})
                self.action = ui.Button("START MISSION", clicked_fn=self._on_action, height=32)
                self.camera_action = ui.Button("CAMERA: FOLLOW (click to orbit manually)", clicked_fn=self._toggle_camera, height=26)
        self.window.visible = True
        self.window.focus()

    def set_callback(self, callback: Any) -> None:
        self._run_callback = callback

    def _toggle_camera(self) -> None:
        self.follow_camera = not self.follow_camera
        self.camera_action.text = "CAMERA: FOLLOW (click to orbit manually)" if self.follow_camera else "CAMERA: MANUAL (click to follow robot)"

    def _on_action(self) -> None:
        if callable(self._run_callback):
            self._run_callback()

    def update(self, pose: Pose2D, *, target: Pose2D, carrying: bool, recoveries: int) -> None:
        phase = "CARRYING" if carrying else "NAVIGATING"
        self.state.text = f"MISSION  {phase}"
        self.target.text = f"TARGET  ({target.x:.1f}, {target.y:.1f})"
        self.pose.text = f"GO2  ({pose.x:.2f}, {pose.y:.2f}, {pose.yaw:.2f})"
        self.payload.text = "PAYLOAD  secured" if carrying else "PAYLOAD  waiting"
        self.recovery.text = f"RECOVERY  {recoveries}"

    def set_ready(self) -> None:
        self.state.text = "MISSION  READY - PRESS PLAY"
        self.action.text = "START MISSION"
        self.action.enabled = True

    def set_running(self) -> None:
        self.state.text = "MISSION  RUNNING"
        self.action.text = "PAUSE MISSION"
        self.action.enabled = True

    def set_paused(self) -> None:
        self.state.text = "MISSION  PAUSED"
        self.action.text = "RESUME MISSION"
        self.action.enabled = True

    def begin(self) -> None:
        self.set_running()

    def complete(self, pose: Pose2D, dropoff: Pose2D, recoveries: int) -> None:
        self.state.text = "MISSION  COMPLETE"
        self.target.text = f"TARGET  ({dropoff.x:.1f}, {dropoff.y:.1f})"
        self.pose.text = f"GO2  ({pose.x:.2f}, {pose.y:.2f}, {pose.yaw:.2f})"
        self.payload.text = "PAYLOAD  released"
        self.recovery.text = f"RECOVERY  {recoveries}"
        self.action.text = "RUN AGAIN"
        self.action.enabled = True

    def failed(self, reason: str, *, retry: bool = True) -> None:
        self.state.text = "MISSION  FAILED"
        self.target.text = reason
        self.action.text = "RUN AGAIN" if retry else "RELAUNCH COMMAND"
        self.action.enabled = retry

    def close(self) -> None:
        self.window.visible = False


def run_mission(
    env: Any,
    policy: Any,
    *,
    pickup: Pose2D,
    dropoff: Pose2D,
    config: BridgeConfig | None = None,
    post_step: Any = None,
    pre_step: Any = None,
    arm: Any = None,
    locate_fn: Any = None,
    home: Pose2D | None = None,
    resolve_delivery: Any = None,
    navigator: Any = None,
    recovery: Any = None,
    source_handler: Any = None,
) -> Any:
    """Run one logistics mission on an already-created Isaac environment."""
    bridge = Go2PolicyBridge(env, policy, config, post_step=post_step, pre_step=pre_step)
    navigate = (lambda point: navigator.navigate(bridge, point)) if navigator is not None else bridge.navigate
    if recovery is not None:
        recovery.bind(bridge, navigate)
    if source_handler is not None:
        source_handler.bind(bridge,navigate)
    if callable(post_step):
        post_step(bridge.pose())
    arm = arm or FixedSceneArm()
    if hasattr(arm, "bind_step"):
        def hold_and_step() -> None:
            bridge._zero_command()
            bridge._policy_step()
        arm.bind_step(hold_and_step)
    route_state = None
    pending_route = []
    aisle_y = (home.y if source_handler is not None and home is not None else pickup.y)-.85
    def navigate_logistics(goal: Pose2D) -> bool:
        nonlocal route_state, pending_route
        if resolve_delivery is not None:
            if route_state != mission.state:
                if mission.state.value == "navigate_dropoff":
                    waypoints = (Pose2D(bridge.pose().x,aisle_y), Pose2D(-6.6,aisle_y),
                                 Pose2D(-6.6,goal.y-.85), Pose2D(goal.x,goal.y-.85))
                elif mission.state.value == "return_home":
                    waypoints = (Pose2D(bridge.pose().x,mission.dropoff.y-.85), Pose2D(-6.6,mission.dropoff.y-.85),
                                 Pose2D(-6.6,aisle_y), Pose2D(goal.x,aisle_y))
                else:
                    waypoints = ()
                pending_route = list((*waypoints,goal))
                route_state = mission.state
            # Retries resume the failed leg, never reconstruct completed legs
            # using the robot's new position as if it were still at the shelf.
            while pending_route:
                point = pending_route[0]
                ok = navigator.navigate(bridge,point,transit=len(pending_route)>1) if navigator is not None else navigate(point)
                if not ok:
                    return False
                pending_route.pop(0)
                if recovery is not None and mission.state.value == 'navigate_dropoff' and len(pending_route) > 1:
                    recovery.inject_after_transit()
            return True
        if home is not None and goal == dropoff:
            # The receiving pedestal lies between the two docking poses.
            # Travel along its clear south aisle, then dock from the west.
            for waypoint in (Pose2D(bridge.pose().x, dropoff.y - .85), Pose2D(dropoff.x, dropoff.y - .85)):
                if not navigate(waypoint):
                    return False
        elif home is not None and goal == home:
            if not navigate(Pose2D(bridge.pose().x, dropoff.y - .85)):
                return False
        return navigate(goal)

    def locate_and_dispatch(object_id):
        if not (locate_fn or (lambda value: bool(value)))(object_id):
            return False
        if resolve_delivery is not None:
            from dataclasses import replace
            code, shelf = resolve_delivery()
            mission.object_id = code
            mission.dropoff = shelf.dock
            geometry = getattr(arm,'detected_geometry',None)
            half_height = geometry['height_m']/2 if geometry is not None else .04
            arm.config = replace(arm.config, dropoff_position=(*shelf.surface[:2],shelf.surface[2]+half_height))
            print(f"SORTING_DISPATCH camera_code={code} shelf={shelf.number} dock={shelf.dock}", flush=True)
        return True

    grasp_attempts = 0
    def grasp_with_fresh_retry(object_id):
        nonlocal grasp_attempts
        # A failed contact attempt may push/rotate the parcel. Never reuse
        # its old geometric target on retry when visual pose mode is enabled.
        if grasp_attempts and getattr(arm,'visual_geometry',False):
            if not locate_and_dispatch(object_id):
                return False
        grasp_attempts += 1
        return arm.grasp(object_id)

    mission = build_mission(
        pose_fn=bridge.pose,
        navigate_fn=navigate_logistics,
        locate_fn=locate_and_dispatch,
        grasp_fn=grasp_with_fresh_retry,
        release_fn=arm.release,
        pickup=pickup,
        dropoff=dropoff,
        home=home,
        state_fn=(lambda state: arm._set_phase(state.value)) if hasattr(arm, "_set_phase") else None,
    )
    mission.recover_drop_fn = recovery
    try:
        result = mission.run(max_steps=30)
    except RuntimeError as exc:
        from go2_logistics.mission import MissionState
        mission.failure_reason = str(exc)
        mission.state = result = MissionState.FAILED
        print(f"PHYSICAL_MISSION failed reason={exc}", flush=True)
    if result.value == "failed" and not mission.failure_reason:
        mission.failure_reason = getattr(arm, "last_error", "") or "Mission retries exhausted"
    print(f"MISSION state={result.value} metrics={mission.metrics}", flush=True)
    mission.navigation = navigator.finish() if navigator is not None else None
    return mission


def run_offline_mission(
    *, pickup: Pose2D, dropoff: Pose2D, failure_rate: float = 0.0, seed: int = 7
) -> Any:
    """Run the same mission contract without Isaac Sim or a GPU.

    This is useful on a laptop while the Go2 USD is being downloaded.  It also
    makes the task-level behavior reproducible in CI; the Isaac runner and this
    fallback share the exact same :class:`MissionController`.
    """
    try:
        from .mission import MissionController
        from .sim import MockArm, MockPerception, MockRobot
    except ImportError:  # Script execution through ``isaaclab.bat -p``.
        from go2_logistics.mission import MissionController
        from go2_logistics.sim import MockArm, MockPerception, MockRobot

    rng = random.Random(seed)
    rate = max(0.0, min(1.0, failure_rate))
    mission = MissionController(
        MockRobot(rng, rate),
        MockPerception(rng, rate / 2.0),
        MockArm(rng, rate),
        pickup,
        dropoff,
    )
    result = mission.run()
    print(f"OFFLINE_MISSION state={result.value} metrics={mission.metrics}", flush=True)
    return mission


def _asset_path(env_cfg: Any) -> str | None:
    """Return the configured robot USD path, if this task exposes one."""
    try:
        return getattr(env_cfg.scene.robot.spawn, "usd_path", None)
    except AttributeError:
        return None


def _asset_is_available(path: str | None) -> bool:
    if not path:
        return True
    if path.startswith(("http://", "https://", "omniverse://")):
        return False
    return Path(path).is_file()


def _bundled_go2_asset() -> Path | None:
    """Prefer the project-local Isaac 6 Go2 bundle when it is present."""
    candidate = Path(__file__).resolve().parents[1] / "artifacts" / "go2_asset" / "go2.usd"
    return candidate if candidate.is_file() else None


def _bundled_franka_asset() -> Path | None:
    """Return the local Franka Panda asset used by the mobile-manipulator scene."""
    candidate = Path(__file__).resolve().parents[1] / "artifacts" / "franka_asset" / "franka.usd"
    return candidate if candidate.is_file() else None


class _VisualPandaFollower:
    """Keep a visual Panda reference mounted on the moving Go2 deck.

    The arm is deliberately visual-only in this showcase mode.  The locomotion
    policy remains the only live articulation; this keeps a late-loaded arm
    asset from changing the trained Go2 dynamics while still making the mobile
    manipulator composition visible and inspectable.
    """

    def __init__(self, root: Any, stage: Any):
        from pxr import UsdGeom

        self.xform = UsdGeom.XformCommonAPI(root)
        try:
            from .visual_gripper import VisualPandaGripper
        except ImportError:
            from go2_logistics.visual_gripper import VisualPandaGripper

        # This marker is a child of the visual-only Panda reference. It follows
        # the same root transform and carries no PhysX schemas.
        self.gripper = VisualPandaGripper(stage, str(root.GetPath()))

    def __call__(self, pose: Pose2D) -> None:
        self.xform.SetTranslate((pose.x + 0.18, pose.y, 0.73))
        self.xform.SetRotate((0.0, 0.0, math.degrees(pose.yaw)))

    def set_state(self, state: str) -> None:
        self.gripper.set_state(state)

    def reset(self) -> None:
        self.gripper.reset()


def _mount_visual_warehouse(scene_path: Path, *, visual_panda: bool = True) -> Any:
    """Reference the project warehouse into the live Isaac Lab stage.

    Isaac Lab owns the Kit stage while its environment is running.  Adding a
    reference after environment construction keeps the policy articulation
    under ``/World/envs`` while making the authored warehouse and camera
    available for a visible policy rollout.
    """
    import omni.usd
    from pxr import Sdf, UsdGeom, UsdPhysics

    if not scene_path.is_file():
        raise FileNotFoundError(f"Warehouse scene not found: {scene_path}")
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac Lab stage is not available while mounting the warehouse")
    root = stage.DefinePrim("/World/Warehouse", "Xform")
    root.GetReferences().ClearReferences()
    root.GetReferences().AddReference(str(scene_path.resolve()))
    UsdGeom.XformCommonAPI(root).SetTranslate((0.0, 0.0, 0.0))

    # Mount the real Panda asset as a visual companion.  It follows the policy
    # robot through the callback below, while its physics schemas stay out of
    # the locomotion contact dynamics.
    panda_asset = _bundled_franka_asset()
    panda_follower = None
    if panda_asset is not None and visual_panda:
        panda_root = stage.DefinePrim("/World/PolicyPanda", "Xform")
        panda_root.GetReferences().ClearReferences()
        panda_root.GetReferences().AddReference(str(panda_asset.resolve()))
        panda_root.SetCustomDataByKey("showcase_asset", "franka_panda")
        panda_root.SetCustomDataByKey("mobile_manipulator_mount", "go2_policy_deck")
        panda_xform = UsdGeom.XformCommonAPI(panda_root)
        panda_xform.SetTranslate((0.18, 0.0, 0.73))
        panda_xform.SetScale((0.62, 0.62, 0.62))
        try:
            stage.Load(panda_root.GetPath())
        except Exception:
            pass
        # References are loaded synchronously by Kit in the visual launcher.
        # Remove physics opinions from this copy: the dedicated physical scene
        # owns the real Panda articulation, while this policy showcase only
        # needs a mesh that can follow the Go2 deck without joint warnings.
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith("/World/PolicyPanda/"):
                continue
            if "Joint" in prim.GetTypeName():
                prim.SetActive(False)
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                try:
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                except Exception:
                    pass
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                try:
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                except Exception:
                    pass
        panda_follower = _VisualPandaFollower(panda_root, stage)
        print(f"POLICY_VISUAL panda={panda_asset.resolve()}", flush=True)
    elif visual_panda:
        print("POLICY_VISUAL panda=missing", flush=True)
    else:
        print("POLICY_VISUAL arm=physical_mounted_articulation", flush=True)

    # Keep the stage's authored cameras available for inspection, but use the
    # Kit Perspective viewport for the live rollout so standard orbit controls
    # remain centered on the scene instead of manipulating a USD camera prim.
    print(f"POLICY_VISUAL warehouse={scene_path.resolve()}", flush=True)
    return panda_follower


def _configure_physical_parcel(env_cfg: Any, pickup: Pose2D, dropoff: Pose2D, ground_bin: bool = False) -> None:
    """Create a graspable rigid parcel and two raised transfer stations."""
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    env_cfg.scene.parcel_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ParcelCamera", update_period=.1,
        height=480, width=640, data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18., horizontal_aperture=20.955, clipping_range=(.05, 10.)),
        offset=CameraCfg.OffsetCfg(pos=(pickup.x, pickup.y, 1.4), rot=(1.,0.,0.,0.), convention="ros"),
    )
    env_cfg.scene.mission_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/MissionCamera", update_period=.2,
        height=600, width=800, data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=20.955, clipping_range=(.05, 100.)),
        offset=CameraCfg.OffsetCfg(pos=(0., -4., 3.), convention="world"),
    )

    env_cfg.scene.parcel = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Parcel",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.035, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, friction_combine_mode="max"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.08, 0.025)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(pickup.x, pickup.y, .061 if ground_bin else .461)),
    )
    from isaaclab_physx.sensors import ContactSensorCfg
    for side in ("left", "right"):
        setattr(env_cfg.scene, f"{side}_grip_contact", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Arm/panda_" + side + "finger",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Parcel"], update_period=0.,
        ))
    for name, point, color in (
        ("pickup_table", pickup, (0.1, 0.35, 0.5)),
        ("dropoff_table", dropoff, (0.1, 0.5, 0.25)),
    ):
        height = .02 if ground_bin and name == "pickup_table" else .42
        setattr(env_cfg.scene, name, AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            spawn=sim_utils.CuboidCfg(
                size=(.28, .28, height) if ground_bin and name == "pickup_table" else (0.16, 0.22, height),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(point.x, point.y, height / 2)),
        ))
    if ground_bin:
        for index, (dx, dy, sx, sy) in enumerate(((0.,-.14,.30,.02), (0.,.14,.30,.02), (-.14,0.,.02,.26), (.14,0.,.02,.26))):
            setattr(env_cfg.scene, f"bin_wall_{index}", AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/BinWall" + str(index),
                spawn=sim_utils.CuboidCfg(size=(sx,sy,.08), collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12,.25,.35))),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(pickup.x+dx,pickup.y+dy,.04)),
            ))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trained Go2 policy on the logistics mission")
    parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-v0")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/go2_official_checkpoint.pt",
    )
    parser.add_argument("--pickup", type=float, nargs=2, default=(4.0, 1.5), metavar=("X", "Y"))
    parser.add_argument("--dropoff", type=float, nargs=2, default=(-3.0, 1.5), metavar=("X", "Y"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument('--run-dir', type=Path, help='Fresh directory for this experiment; existing paths are rejected')
    parser.add_argument("--grasp-mode", choices=("assisted", "contact"), default="assisted")
    parser.add_argument("--ground-bin", action="store_true", help="Experimental ground-bin pickup with 8 cm rim")
    parser.add_argument("--sorting", action="store_true", help="Visual code dispatch to 30 shelves; single-parcel stage")
    parser.add_argument("--obstacles", action="store_true", help="Unknown aisle blocker and online range-sensor A* navigation")
    parser.add_argument('--conveyor',action='store_true',help='Contact-driven belt feeding the ground bin')
    parser.add_argument('--warehouse-demo',action='store_true',help='Integrated warehouse: scene, conveyor, occlusion, sensed navigation and two deliveries')
    parser.add_argument('--dynamic-obstacle',action='store_true',help='Moving aisle fixture and observed-free map clearing')
    parser.add_argument('--continuous',action='store_true',help='Warehouse finite multi-order line with automatic receiving backpressure')
    parser.add_argument('--streaming-feed',action='store_true',help='Fleet-only finite 6-10 parcel physical queue with per-item RGB-D dispatch audit')
    parser.add_argument('--fleet',type=int,choices=(0,2),default=0,help='Two dogs in one shared warehouse and one physics clock')
    parser.add_argument('--arm-model', choices=('panda_surrogate','z1'), default='panda_surrogate',
                        help='Opt-in original-mass Z1 warehouse backend; fleet only, experimental')
    parser.add_argument('--fleet-smoke',action='store_true',help='Shared-world parallel navigation smoke check, without deliveries')
    parser.add_argument('--fleet-gui-check',action='store_true',help='Native visual callback/physics-freeze diagnostic; exits after smoke route')
    parser.add_argument('--fleet-endurance-cycles',type=int,default=0,help='Repeat shared-world rectangular unloaded routes; diagnostic only, no deliveries')
    parser.add_argument('--fleet-ticks',type=int,default=30000,help='Bounded shared-world control tick budget')
    parser.add_argument('--indoor-warehouse',action='store_true',help='Locally cached NVIDIA indoor warehouse shell')
    parser.add_argument('--recover-drops',action='store_true',help='Experimental local RGB-D recovery of upright floor parcels')
    parser.add_argument('--inject-drop',action='store_true',help='Open gripper once after the first loaded transit, enabling drop recovery')
    parser.add_argument('--visual-geometry',action='store_true',help='Estimate known cuboid face pose from rendered RGB-D')
    parser.add_argument('--wide-gripper',action='store_true',help='Experimental custom 5cm stroke parallel pads; not stock Panda')
    parser.add_argument('--occlusion-demo',action='store_true',help='Physically stage an upper parcel, deliver the exposed lower parcel, then retrieve the staged one')
    parser.add_argument('--parcel-roll',type=float,default=0.,help='Initial parcel roll in degrees; geometry experiment only')
    parser.add_argument('--parcel-yaw',type=float,default=0.,help='Initial parcel yaw in degrees; geometry experiment only')
    parser.add_argument('--batch-codes',default='',help='Comma-separated distinct conveyor labels; defaults to parcel-code')
    parser.add_argument("--parcel-code", default="1-001", help="Label content for scene generation, not a navigation goal")
    parser.add_argument("--locomotion-only", action="store_true", help="Diagnostic: run Go2 without the mounted physical arm")
    parser.add_argument("--wait-for-start", action="store_true", help="Wait for Play instead of starting the visual mission automatically")
    parser.add_argument("--offline", action="store_true", help="Run the deterministic mission without Isaac Sim")
    parser.add_argument(
        "--allow-remote-assets",
        action="store_true",
        help="Allow Isaac Sim to resolve the Go2 USD from Nucleus (may block when offline)",
    )
    parser.add_argument("--failure-rate", type=float, default=0.0, help="Failure rate for --offline recovery testing")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Show the PhysX policy rollout and mount the project warehouse USD",
    )
    parser.add_argument(
        "--warehouse-scene",
        type=Path,
        default=Path("artifacts/go2_warehouse.usd"),
        help="Warehouse USD to reference when --visual is enabled",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    # Parse the task-only arguments before importing Isaac.  ``--offline`` is
    # therefore genuinely lightweight and works in a plain Python environment.
    early_args, _ = parser.parse_known_args(argv)
    pickup = Pose2D(*early_args.pickup)
    dropoff = Pose2D(*early_args.dropoff)
    if early_args.offline:
        run_offline_mission(
            pickup=pickup,
            dropoff=dropoff,
            failure_rate=early_args.failure_rate,
            seed=early_args.seed,
        )
        return
    # Imports are delayed so `python -m pytest go2_logistics` remains usable on
    # machines that do not have Isaac Sim installed.
    import gymnasium as gym
    import importlib.metadata as metadata
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config
    from rsl_rl.runners import OnPolicyRunner

    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.fleet_gui_check:
        if not args.visual or args.fleet_endurance_cycles:
            parser.error('Fleet GUI check requires --visual and excludes endurance cycles')
        args.fleet_smoke=True
        args.wait_for_start=True
    if args.fleet_endurance_cycles<0:
        parser.error('Fleet endurance cycles must be nonnegative')
    if args.fleet_endurance_cycles:
        args.fleet_smoke=True
    if args.indoor_warehouse and not args.fleet and not args.fleet_smoke and not args.continuous:
        args.warehouse_demo=True
    if args.fleet_smoke:
        args.fleet=2
    if args.arm_model == 'z1':
        if not args.fleet or args.recover_drops or args.inject_drop or args.wide_gripper:
            parser.error('Experimental Z1 backend requires a fleet; its drop-recovery adapter is not yet connected')
        args.device = 'cpu'
        args.grasp_mode = 'contact'
    if args.fleet:
        args.continuous=True
        if not args.batch_codes:
            args.batch_codes='1-001,23-002'
        if args.locomotion_only or args.occlusion_demo or args.visual_geometry or args.wide_gripper or args.dynamic_obstacle:
            parser.error('Fleet supports conveyor contact sorting and bounded drop recovery; diagnostic, occlusion, geometry and scripted dynamic-obstacle modes remain single-robot experiments')
        if args.fleet_smoke and (args.recover_drops or args.inject_drop):
            parser.error('Fleet drop recovery requires a delivery mission, not navigation smoke')
    if args.streaming_feed and (not args.fleet or args.fleet_smoke):
        parser.error('--streaming-feed requires a two-robot fleet delivery mission')
    if args.dynamic_obstacle:
        args.obstacles=True
    if args.warehouse_demo:
        args.sorting=args.ground_bin=args.obstacles=args.conveyor=args.occlusion_demo=True
        args.grasp_mode='contact'
    if args.continuous:
        args.warehouse_demo=args.sorting=args.ground_bin=args.obstacles=args.conveyor=True
        args.occlusion_demo=False
        args.grasp_mode='contact'
        if not args.batch_codes:
            args.batch_codes='1-001,12-002,23-003'
    args.recover_drops = args.recover_drops or args.inject_drop
    args.wide_gripper = args.wide_gripper or abs(args.parcel_roll % 180) > 1
    if args.occlusion_demo:
        args.visual_geometry=True
        if args.recover_drops or args.parcel_roll or args.parcel_yaw:
            parser.error('Occlusion baseline requires upright parcels; omit drop/orientation flags')
    if args.visual_geometry and (not args.sorting or not args.ground_bin or args.grasp_mode != 'contact'):
        parser.error('Visual geometry requires --sorting --ground-bin --grasp-mode contact')
    if args.wide_gripper and (not args.visual_geometry or args.occlusion_demo):
        parser.error('Wide gripper is a separate --visual-geometry experiment, not the occlusion baseline')
    if (args.parcel_roll or args.parcel_yaw) and not args.visual_geometry:
        parser.error('Initial orientation experiments require --visual-geometry')
    if args.recover_drops and (not args.sorting or not args.ground_bin or args.grasp_mode != 'contact' or args.locomotion_only):
        parser.error('Drop recovery requires --sorting --ground-bin --grasp-mode contact')
    if args.conveyor and (not args.sorting or not args.ground_bin or args.grasp_mode!='contact' or args.locomotion_only):
        parser.error('Conveyor currently requires --sorting --ground-bin --grasp-mode contact')
    batch_codes = args.batch_codes.split(',') if args.batch_codes else ([args.parcel_code,'23-002'] if args.occlusion_demo else [args.parcel_code])
    batch_enabled = args.conveyor or args.occlusion_demo
    if args.streaming_feed:
        from go2_logistics.streaming_feed import MIN_STREAM_CODES, MAX_STREAM_CODES
        if not MIN_STREAM_CODES<=len(batch_codes)<=MAX_STREAM_CODES:
            parser.error(f'--streaming-feed requires {MIN_STREAM_CODES}..{MAX_STREAM_CODES} unique batch codes')
    elif args.continuous and not 1<=len(batch_codes)<=4:
        parser.error('Current warehouse belt holds 1..4 preloaded orders; use --streaming-feed for the bounded 6..10 parcel fleet recipe')
    if batch_enabled:
        args.parcel_code=batch_codes[0]
        from go2_logistics.sorting_layout import ParcelCode
        for code in batch_codes:
            ParcelCode.parse(code)
        if len(batch_codes)!=len(set(batch_codes)):
            parser.error('Batch labels must be unique')
    # A policy rollout is intended to be headless in batch/CI usage.  The
    # standard Isaac Lab launcher arguments still remain available to override
    # device and rendering options.
    args.headless = not args.visual
    args.enable_cameras = not args.locomotion_only
    env_cfg_seed = args.seed
    # ``resolve_task_config`` runs Hydra and reads ``sys.argv`` itself.  Strip
    # the runner's already-consumed flags so Hydra does not try to parse them as
    # configuration overrides (the official Isaac Lab play scripts do the same
    # through ``setup_preset_cli``).
    sys.argv = [sys.argv[0]]
    # Warp's default Windows cache can be owned by another Isaac process. Keep
    # this short rollout isolated in the project artifacts directory so a stale
    # or locked global cache cannot abort environment construction.
    from go2_logistics.run_isolation import create_run_directory
    artifact_root = Path(__file__).resolve().parents[1] / 'artifacts'
    args.run_dir = create_run_directory(artifact_root / ('fleet' if args.fleet else 'missions'), args.run_dir)
    print(f'EXPERIMENT_RUN_DIR {args.run_dir}', flush=True)
    warp_cache = args.run_dir / 'warp_cache'
    warp_cache.mkdir(parents=True, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(warp_cache)

    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
    from go2_logistics.local_ground import configure_local_ground, configure_local_sky
    configure_local_ground(env_cfg, allow_remote=args.allow_remote_assets)
    configure_local_sky(env_cfg, allow_remote=args.allow_remote_assets)
    # Isaac Lab's stock config points to a remote Nucleus URL.  The project
    # bundles the same Isaac 6 Go2 asset locally, so policy runs are repeatable
    # offline and do not depend on Nucleus resolution.
    bundled_asset = _bundled_go2_asset()
    if bundled_asset is not None:
        env_cfg.scene.robot.spawn.usd_path = str(bundled_asset)
        # The Isaac 6 Go2 bundle is a variant USD.  Its default Physics=None
        # selection intentionally exposes only the render meshes; Isaac Lab
        # needs the physx variant to discover the articulation and rigid bodies
        # used by contact sensors and the locomotion controller.
        if hasattr(env_cfg.scene.robot.spawn, "variants"):
            env_cfg.scene.robot.spawn.variants = {"Physics": "physx"}
        print(f"Using bundled Go2 USD: {bundled_asset}", flush=True)
        print("Selecting Go2 USD variants: Physics=physx", flush=True)
    usd_path = _asset_path(env_cfg)
    if not args.allow_remote_assets and not _asset_is_available(usd_path):
        print(
            "Go2 USD is not available locally: "
            f"{usd_path}\n"
            "Use --offline for the task-level demo, or download/cache the USD "
            "and rerun with --allow-remote-assets.",
            file=sys.stderr,
        )
        return
    env_cfg.scene.num_envs = 1
    # Opt-in diagnostic: apply external forces on each TGS position iteration.
    # Keep the established baseline until a full physical run validates it.
    if os.environ.get('GO2_TGS_EXTERNAL_FORCES') == '1':
        env_cfg.sim.physics.enable_external_forces_every_iteration = True
        print('PHYSICS_DIAGNOSTIC TGS external forces every iteration enabled', flush=True)
    if args.obstacles:
        from go2_logistics.sensor_navigation import configure_navigation_obstacles
        env_cfg.sim.enable_scene_query_support = True
        configure_navigation_obstacles(env_cfg)
    env_cfg.seed = env_cfg_seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_base.params["velocity_range"] = {}
    # A delivery includes arm motion and two navigation legs. The training
    # task's 20 s timeout would silently reset the robot mid-delivery.
    env_cfg.episode_length_s = max(300.0, args.steps * env_cfg.decimation * env_cfg.sim.dt * 8)
    env_cfg.commands.base_velocity.debug_vis = False
    # The stock velocity task randomizes the reset pose over a half-meter and
    # a full turn.  The warehouse mission has a fixed spawn transform and its
    # waypoint goals are authored in that frame, so keep the initial pose
    # deterministic while retaining the trained locomotion policy.
    if not os.environ.get("GO2_RANDOM_RESET"):
        try:
            reset_pose = env_cfg.events.reset_base.params["pose_range"]
            reset_pose["x"] = (0.0, 0.0)
            reset_pose["y"] = (0.0, 0.0)
            reset_pose["yaw"] = (0.0, 0.0)
        except (AttributeError, KeyError, TypeError):
            pass
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    physical_arm_enabled = not args.locomotion_only
    if physical_arm_enabled:
        from go2_logistics.mounted_robot_asset import configure_robot

        mounted_asset = args.run_dir / ("go2_ground_panda.usda" if args.ground_bin else "go2_mounted_panda.usda")
        if args.wide_gripper:
            mounted_asset = args.run_dir / 'go2_ground_panda_wide.usda'
        if args.arm_model != 'z1':
            configure_robot(env_cfg, mounted_asset)
        _configure_physical_parcel(env_cfg, pickup, dropoff, ground_bin=args.ground_bin)
        if args.arm_model == 'z1':
            from go2_logistics.z1_warehouse_backend import configure_z1_warehouse
            mounted_asset = args.run_dir/'go2_z1.usda'
            configure_z1_warehouse(env_cfg, mounted_asset)
            agent_cfg.device = 'cpu'
            env_cfg.sim.device = 'cpu'
        if args.sorting:
            from go2_logistics.sorting_layout import ParcelCode
            from go2_logistics.sorting_scene import configure_sorting_scene
            ParcelCode.parse(args.parcel_code)
            configure_sorting_scene(env_cfg, pickup, args.ground_bin)
        if args.recover_drops:
            from go2_logistics.drop_recovery import configure_recovery_camera
            configure_recovery_camera(env_cfg)
        if args.conveyor and not args.occlusion_demo:
            from go2_logistics.conveyor import configure_conveyor
            configure_conveyor(env_cfg,pickup,batch_codes,streaming=args.streaming_feed)
            args.device='cpu'
            agent_cfg.device='cpu'
            env_cfg.episode_length_s=max(env_cfg.episode_length_s,300.*len(batch_codes))
        if args.visual_geometry:
            from scipy.spatial.transform import Rotation
            if args.wide_gripper:
                env_cfg.scene.parcel.spawn.collision_props.contact_offset=.001
                env_cfg.scene.parcel.spawn.collision_props.rest_offset=0.
            orientation = Rotation.from_euler('xz',[args.parcel_roll,args.parcel_yaw],degrees=True).as_quat()
            env_cfg.scene.parcel.init_state.rot = tuple(float(v) for v in orientation)
            x,y,z = env_cfg.scene.parcel.init_state.pos
            # Initial height is scene authoring only; physics controls all
            # poses after reset. Conservative clearance lets the parcel settle.
            env_cfg.scene.parcel.init_state.pos = (x,y,max(z,.09))
            args.device = agent_cfg.device = env_cfg.sim.device = 'cpu'
        if args.occlusion_demo:
            from go2_logistics.occlusion import configure_occlusion
            configure_occlusion(env_cfg,pickup,batch_codes)
            if args.conveyor:
                from go2_logistics.conveyor import configure_conveyor
                configure_conveyor(env_cfg,pickup,batch_codes,stack_feed=True)
            env_cfg.episode_length_s=max(env_cfg.episode_length_s,600.)
    if args.warehouse_demo:
        from go2_logistics.warehouse_scene import configure_warehouse
        configure_warehouse(env_cfg)
    if args.fleet:
        from go2_logistics.fleet_runtime import configure_fleet
        configure_fleet(env_cfg,args.fleet)
        env_cfg.episode_length_s=max(env_cfg.episode_length_s,args.fleet_ticks*env_cfg.decimation*env_cfg.sim.dt+60.)
        if not args.visual:
            # Range navigation/control still run at 50 Hz. RTX cameras need
            # at most 10 Hz here; perception explicitly renders on capture.
            env_cfg.sim.render_interval=20 if args.arm_model=='z1' else env_cfg.decimation*5
    if args.indoor_warehouse:
        from go2_logistics.indoor_warehouse import configure_indoor_warehouse
        configure_indoor_warehouse(env_cfg)
    if args.dynamic_obstacle:
        env_cfg.scene.aisle_blocker.init_state.pos=(-1.,-1.35,.375)
    from go2_logistics.physics_resolution import refine_physics_timestep
    timing=refine_physics_timestep(env_cfg,int(os.environ.get('GO2_PHYSICS_SUBSTEPS','1')))
    print(f'PHYSICS_TIMING {timing}',flush=True)
    with launch_simulation(env_cfg, args):
        if physical_arm_enabled:
            from go2_logistics.mounted_robot_asset import ensure_mounted_robot_asset
            if args.arm_model == 'z1':
                from go2_logistics.z1_warehouse_backend import ensure_z1_warehouse_asset
                ensure_z1_warehouse_asset(mounted_asset)
            else:
                ensure_mounted_robot_asset(ground_reach=args.ground_bin,wide_gripper=args.wide_gripper,output_path=mounted_asset)
        env = gym.make(args.task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
        return_policy = None
        return_checkpoint = os.environ.get('GO2_Z1_RETURN_CHECKPOINT')
        if args.fleet and args.arm_model=='z1' and return_checkpoint:
            import copy
            import torch
            return_policy=copy.deepcopy(policy)
            weights=torch.load(return_checkpoint,weights_only=False,map_location=wrapped.unwrapped.device)
            return_policy.load_state_dict(weights['actor_state_dict'],strict=True)
            return_policy.eval()
        wrapped.reset()
        if args.warehouse_demo:
            import omni.usd
            from go2_logistics.warehouse_scene import author_warehouse
            author_warehouse(omni.usd.get_context().get_stage(),args.run_dir/'warehouse_signs')
        if args.sorting:
            import omni.usd
            from go2_logistics.sorting_scene import author_sorting_labels
            author_sorting_labels(omni.usd.get_context().get_stage(), args.parcel_code, args.run_dir/'sorting_labels')
            if batch_enabled:
                from go2_logistics.conveyor import author_conveyor
                author_conveyor(omni.usd.get_context().get_stage(),batch_codes,args.run_dir/'sorting_labels')
            if args.visual_geometry:
                from go2_logistics.sorting_scene import author_parcel_label
                for index,code in enumerate(batch_codes if batch_enabled else [args.parcel_code]):
                    author_parcel_label(omni.usd.get_context().get_stage(),code,args.run_dir/'sorting_labels',
                                       '/World/envs/env_0/Parcel'+(str(index) if index else ''),all_faces=True)
        if args.indoor_warehouse:
            from pxr import UsdGeom
            ground=omni.usd.get_context().get_stage().GetPrimAtPath(env_cfg.scene.terrain.prim_path)
            if ground.IsValid():
                # Keep ground collision while rendering only the official floor.
                UsdGeom.Imageable(ground).MakeInvisible()
        if args.fleet:
            from go2_logistics.fleet_runtime import run_shared_fleet
            try:
                run_shared_fleet(wrapped,policy,args,pickup,batch_codes,return_policy=return_policy)
            finally:
                wrapped.close()
            return
        panda_follower = None
        visual_payload = None
        visual_step = None
        policy_hud = None
        policy_timeline = None
        policy_run_requested = {"value": False}
        policy_mission_active = {"value": False}
        physical_arm = None
        perception = None
        recorder = None
        if physical_arm_enabled:
            import omni.usd
            from go2_logistics.physical_manipulation import PhysicalPandaArm, PhysicalPandaConfig

            physical_arm = PhysicalPandaArm(
                wrapped,
                omni.usd.get_context().get_stage(),
                wrapped.unwrapped.scene["parcel"],
                box_path="/World/envs/env_0/Parcel",
                hand_path="/World/envs/env_0/Robot/Arm/panda_hand",
                config=PhysicalPandaConfig(dropoff_position=(dropoff.x, dropoff.y, 0.46), grasp_mode=args.grasp_mode,
                                           finger_open=.049 if args.wide_gripper else .024,
                                           reset_before_grasp=batch_enabled or args.visual_geometry,
                                           stow_after_release=args.occlusion_demo or args.continuous,
                                           grasp_height_offset=.012, lift_height=.50 if args.ground_bin else .15,
                                           tool_offset=(0.,0.,.1034*(.85 if args.ground_bin else .62))),
            )
            from go2_logistics.camera_perception import RedParcelPerception
            perception = RedParcelPerception(wrapped, physical_arm, Path("artifacts/physical_mission"), decode_code=args.sorting)
            perception.estimate_geometry = args.visual_geometry
            physical_arm.visual_geometry = args.visual_geometry
            from go2_logistics.mission_recorder import MissionRecorder
            recorder = MissionRecorder(wrapped, physical_arm, Path("artifacts/physical_mission"))
            if args.continuous:
                from go2_logistics.conveyor_control import ReceivingGate
                recorder.receiving_gate=ReceivingGate(wrapped,pickup)
            if args.dynamic_obstacle:
                from go2_logistics.dynamic_obstacle import CrossingObstacle
                recorder.dynamic_obstacle=CrossingObstacle(wrapped)
            if batch_enabled:
                from go2_logistics.conveyor import ConveyorBatch
                if args.occlusion_demo:
                    from go2_logistics.occlusion import OcclusionBatch
                    batch=OcclusionBatch(wrapped,physical_arm,batch_codes,pickup,perception,stack_feed=args.conveyor)
                else:
                    batch=ConveyorBatch(wrapped,physical_arm,batch_codes)
                    if args.continuous:
                        batch.mode='finite_order_line_with_receiving_backpressure'
                recorder.batch=batch
                perception.bind_visible=batch.bind_visible
                if hasattr(batch, 'bind_perception'):
                    batch.bind_perception(perception)
        if not args.visual and physical_arm_enabled and not args.sorting:
            _mount_visual_warehouse(args.warehouse_scene.resolve(), visual_panda=False)
            from pxr import UsdGeom
            old_box = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Warehouse/Task/PickupBox")
            if old_box.IsValid():
                UsdGeom.Imageable(old_box).MakeInvisible()
        if args.visual:
            panda_follower = None if args.sorting else _mount_visual_warehouse(args.warehouse_scene.resolve(), visual_panda=False)
            import omni.usd
            import omni.timeline
            import omni.kit.app

            visual_stage = omni.usd.get_context().get_stage()
            policy_timeline = omni.timeline.get_timeline_interface()
            # STOP destroys the PhysX tensor views. Keep the initialized scene
            # alive while the viewport and controls are being constructed.
            env.unwrapped.sim.pause()
            app = omni.kit.app.get_app()
            env.unwrapped.sim.set_camera_view(eye=(0.0, -8.0, 8.0), target=(0.0, 1.0, 0.6))
            try:
                import omni.kit.viewport.utility as viewport_utility
                from pxr import Sdf

                viewport = viewport_utility.get_active_viewport()
                if viewport is not None:
                    viewport.camera_path = Sdf.Path("/OmniverseKit_Persp")
            except Exception as exc:
                print(f"POLICY_VISUAL perspective_camera=fallback reason={exc}", flush=True)
            print("POLICY_VISUAL camera=/OmniverseKit_Persp controls=standard_orbit", flush=True)
            # The physical parcel is owned by PhysX; hide the old display box.
            if physical_arm_enabled:
                from pxr import UsdGeom
                old_box = visual_stage.GetPrimAtPath("/World/Warehouse/Task/PickupBox")
                if old_box.IsValid():
                    UsdGeom.Imageable(old_box).MakeInvisible()
                visual_payload = physical_arm
            else:
                visual_payload = VisualPayloadAdapter(visual_stage, pickup, dropoff)
            policy_hud = PolicyMissionHud()

            def wait_for_timeline_play() -> None:
                paused = False
                while app.is_running() and not policy_timeline.is_playing():
                    if policy_timeline.is_stopped():
                        raise RuntimeError("Timeline stopped; relaunch the command to rebuild physics")
                    if policy_hud is not None:
                        policy_hud.set_paused()
                    paused = True
                    app.update()
                if not app.is_running():
                    raise SystemExit(0)
                if paused and policy_hud is not None and policy_mission_active["value"]:
                    # Also refresh Fabric after a GUI resume; otherwise the
                    # articulation meshes can remain frozen at their old pose.
                    env.unwrapped.sim.play()
                    policy_hud.set_running()

            def request_policy_mission() -> None:
                # UI callbacks must not recursively pump app.update().
                if policy_mission_active["value"]:
                    if policy_timeline.is_playing():
                        policy_timeline.pause()
                    else:
                        policy_timeline.play()
                else:
                    policy_run_requested["value"] = True

            policy_hud.set_callback(request_policy_mission)
            policy_hud.set_ready()
            policy_run_requested["value"] = not args.wait_for_start
            print(f"POLICY_VISUAL_READY autostart={not args.wait_for_start}", flush=True)

            def visual_step(pose: Pose2D) -> None:
                if callable(panda_follower):
                    panda_follower(pose)
                visual_payload.sync(pose)
                if recorder is not None:
                    recorder.step(pose)
                if policy_hud is not None:
                    if policy_hud.follow_camera:
                        root = _tensor(env.unwrapped.scene["robot"].data.root_pos_w)[0]
                        rx, ry, rz = (float(value) for value in root)
                        env.unwrapped.sim.set_camera_view(eye=(rx+1.0, ry-2.2, rz+1.5), target=(rx+.25, ry, rz+.25))
                    target = Pose2D(0., 0.) if physical_arm is not None and physical_arm.state == "return_home" else (dropoff if visual_payload.held else pickup)
                    policy_hud.update(
                        pose,
                        target=target,
                        carrying=visual_payload.held,
                        recoveries=0,
                    )
                    if physical_arm is not None:
                        policy_hud.state.text = "MISSION  " + physical_arm.state.upper()

        bridge_config = BridgeConfig(
            max_control_steps=max(args.steps, 1800 if args.obstacles else 1200) if args.sorting else args.steps,
            position_tolerance=(0.04 if batch_enabled or args.visual_geometry else 0.07) if physical_arm_enabled else 0.35,
        )

        def execute_mission() -> Any:
            policy_mission_active["value"] = True
            if physical_arm is not None:
                physical_arm.disable_constraint()
            if policy_hud is not None:
                policy_hud.begin()
                env.unwrapped.sim.play()
            wrapped.reset()
            if recorder is not None:
                if recorder.dynamic_obstacle is not None:
                    recorder.dynamic_obstacle.elapsed=None
                    recorder.dynamic_obstacle.events=[]
                    recorder.dynamic_obstacle.step_count=0
                if recorder.receiving_gate is not None:
                    recorder.receiving_gate.reset()
            mission_arm = physical_arm or visual_payload
            def resolve_sorting_delivery():
                from go2_logistics.sorting_layout import shelf_for_code
                code = perception.last_detection["parcel_code"]
                return code, shelf_for_code(code)
            if batch_enabled:
                batch.completed.clear()
                batch.runs.clear()
            for cycle in range(len(batch_codes) if batch_enabled else 1):
                if mission_arm is not None:
                    mission_arm.prepare()
                if recorder is not None:
                    recorder.begin()
                    perception.output_dir = recorder.output_dir
                    perception.last_detection = None
                if args.occlusion_demo:
                    batch.begin_cycle(cycle,recorder.output_dir)
                navigator = None
                if args.obstacles:
                    from go2_logistics.sensor_navigation import SensorNavigator
                    navigator = SensorNavigator(wrapped, recorder.output_dir if recorder else None,dynamic_map=args.dynamic_obstacle)
                if recorder is not None:
                    recorder.navigator=navigator
                print(f'BATCH_CYCLE index={cycle+1} world_reset={cycle==0}',flush=True)
                recovery = None
                if args.recover_drops:
                    from go2_logistics.drop_recovery import DroppedParcelRecovery
                    recovery = DroppedParcelRecovery(wrapped, mission_arm, recorder.output_dir,
                                                     inject=args.inject_drop and cycle == 0)
                if recorder is not None:
                    recorder.recovery = recovery
                mission_result = run_mission(
                    wrapped,
                    policy,
                    pickup=batch.source_dock() if args.occlusion_demo else (Pose2D(pickup.x - (0.58 if batch_enabled or args.visual_geometry else 0.50), pickup.y) if physical_arm_enabled else pickup),
                    dropoff=Pose2D(dropoff.x - 0.50, dropoff.y) if physical_arm_enabled else dropoff,
                    config=bridge_config,
                    post_step=visual_step or (recorder.step if recorder is not None else panda_follower),
                    pre_step=wait_for_timeline_play if args.visual else None,
                    arm=mission_arm,
                    locate_fn=(batch.locate if hasattr(batch, 'locate') else perception.locate) if batch_enabled else (perception.locate if perception is not None else None),
                    home=(Pose2D(pickup.x-(.58 if batch_enabled or args.visual_geometry else .50),pickup.y) if args.sorting else Pose2D(0., 0.)) if physical_arm_enabled else None,
                    resolve_delivery=resolve_sorting_delivery if args.sorting else None,
                    navigator=navigator,
                    recovery=recovery,
                    source_handler=batch if args.occlusion_demo else None,
                )
                if recorder is not None:
                    recorder.finish(mission_result, perception.last_detection)
                if batch_enabled:
                    batch.finish_cycle(mission_result,recorder.output_dir)
                if mission_result.state.value!='complete':
                    break
            if args.visual and policy_hud is not None:
                # Pause preserves both the final pose and physics handles for
                # replay. Report the actual mission result, including failure.
                env.unwrapped.sim.pause()
                if mission_result.state.value == "complete":
                    policy_hud.complete(
                        mission_result.robot.pose(), dropoff, mission_result.metrics.recoveries
                    )
                else:
                    policy_hud.failed("Task failed; see terminal for the failed phase")
                print(f"POLICY_VISUAL result={mission_result.state.value}; use RUN AGAIN to replay.", flush=True)
            policy_mission_active["value"] = False
            return mission_result

        if args.visual:
            app.update()
            while app.is_running():
                app.update()
                if policy_timeline.is_stopped():
                    policy_hud.failed("Timeline stopped; relaunch this command", retry=False)
                    continue
                if policy_run_requested["value"] or policy_timeline.is_playing():
                    policy_run_requested["value"] = False
                    try:
                        mission = execute_mission()
                    except (RuntimeError, AttributeError) as exc:
                        policy_mission_active["value"] = False
                        policy_hud.failed(str(exc), retry=False)
                        print(f"POLICY_VISUAL_ERROR {exc}", flush=True)
                        # Leave the error visible without retrying invalid
                        # tensor handles every frame.
                        while app.is_running():
                            app.update()
        else:
            mission = execute_mission()
            if mission.state.value != "complete":
                raise RuntimeError(mission.failure_reason or "Physical mission failed")
        wrapped.close()


if __name__ == "__main__":
    main()

