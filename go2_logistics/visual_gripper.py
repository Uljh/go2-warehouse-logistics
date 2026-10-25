"""Physics-free visual Panda gripper state marker used by the demos.

The policy and physical showcase scenes deliberately keep the arm outside the
Go2 locomotion articulation. This helper adds a small, clearly visible
end-effector proxy under the existing arm/base prim and changes its finger
spacing and color at grasp/release transitions. It contains no collision,
rigid-body, or joint opinions, so it cannot alter the simulated dynamics.
"""

from __future__ import annotations

from typing import Any


class VisualPandaGripper:
    """Render a stateful parallel gripper without touching PhysX."""

    _STATES = {
        "waiting": ((0.052, 0.055, 0.070), (0.05, 0.30, 0.80)),
        "grasp": ((0.018, 0.022, 0.035), (1.00, 0.58, 0.04)),
        "carried": ((0.018, 0.022, 0.035), (1.00, 0.78, 0.08)),
        "release": ((0.052, 0.055, 0.070), (1.00, 0.58, 0.04)),
        "released": ((0.052, 0.055, 0.070), (0.10, 0.82, 0.38)),
    }

    def __init__(
        self,
        stage: Any,
        parent_path: str,
        *,
        name: str = "ShowcaseGripper",
        local_translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_scale: float = 1.0,
    ):
        from pxr import Gf, UsdGeom

        self.Gf = Gf
        self.UsdGeom = UsdGeom
        self.stage = stage
        self.path = f"{parent_path.rstrip('/')}/{name}"
        self.root = UsdGeom.Xform.Define(stage, self.path).GetPrim()
        self.root.SetCustomDataByKey("visual_only", True)
        self.root.SetCustomDataByKey("manipulator_component", "panda_gripper")
        self.xform = UsdGeom.XformCommonAPI(self.root)
        self.xform.SetTranslate(Gf.Vec3d(*local_translate))
        self.xform.SetScale(Gf.Vec3f(local_scale, local_scale, local_scale))

        # Local coordinates match the bundled Panda asset's hand region:
        # +X is forward, +/-Y are the two fingers, and Z is vertical.
        self.palm = self._cube("Palm", (0.18, 0.16, 0.08), (0.10, 0.0, 0.90))
        self.left = self._cube("LeftFinger", (0.09, 0.035, 0.14), (0.13, 0.05, 0.83))
        self.right = self._cube("RightFinger", (0.09, 0.035, 0.14), (0.13, -0.05, 0.83))
        self.tip = self._cube("ContactTip", (0.025, 0.12, 0.025), (0.18, 0.0, 0.78))
        self._set_state("waiting")

    def _cube(self, name: str, size: tuple[float, float, float], xyz: tuple[float, float, float]) -> Any:
        from pxr import Sdf

        prim = self.UsdGeom.Cube.Define(self.stage, Sdf.Path(f"{self.path}/{name}"))
        prim.CreateSizeAttr(1.0)
        prim.AddScaleOp().Set(self.Gf.Vec3f(*size))
        self.UsdGeom.XformCommonAPI(prim).SetTranslate(self.Gf.Vec3d(*xyz))
        # No physics APIs are authored on these display-only cubes.
        return prim

    def _set_color(self, color: tuple[float, float, float]) -> None:
        for part in (self.palm, self.left, self.right, self.tip):
            part.CreateDisplayColorAttr([self.Gf.Vec3f(*color)])

    def _set_state(self, state: str) -> None:
        spacing, color = self._STATES.get(state, self._STATES["waiting"])
        self.state = state
        self.root.SetCustomDataByKey("gripper_state", state)
        # Finger motion is an explicit open/close animation at the task
        # transition boundary; the surrounding robot and arm are untouched.
        self.UsdGeom.XformCommonAPI(self.left).SetTranslate(
            self.Gf.Vec3d(0.13, spacing[0], 0.83)
        )
        self.UsdGeom.XformCommonAPI(self.right).SetTranslate(
            self.Gf.Vec3d(0.13, -spacing[0], 0.83)
        )
        self.UsdGeom.XformCommonAPI(self.tip).SetTranslate(
            self.Gf.Vec3d(0.18, 0.0, 0.78 if state in ("grasp", "carried") else 0.80)
        )
        self._set_color(color)

    def set_state(self, state: str) -> None:
        """Set one of waiting, grasp, carried, release, or released."""
        self._set_state(state)

    def reset(self) -> None:
        self._set_state("waiting")
