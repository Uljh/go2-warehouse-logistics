"""Numpy-only kinematics for the official Unitree Z1 URDF chain."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _xyz(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.zeros(3, dtype=float)
    return np.asarray([float(v) for v in element.attrib.get("xyz", "0 0 0").split()], dtype=float)


def _rpy(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.zeros(3, dtype=float)
    return np.asarray([float(v) for v in element.attrib.get("rpy", "0 0 0").split()], dtype=float)


def _rot_rpy(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s, one = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([[c + x*x*one, x*y*one - z*s, x*z*one + y*s], [y*x*one + z*s, c + y*y*one, y*z*one - x*s], [z*x*one - y*s, z*x*one + x*s, c + z*z*one]])


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = _rot_rpy(rpy)
    out[:3, 3] = xyz
    return out


def _rotvec(R: np.ndarray) -> np.ndarray:
    value = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(value))
    if angle < 1e-8:
        return np.array([(R[2, 1] - R[1, 2]) / 2.0, (R[0, 2] - R[2, 0]) / 2.0, (R[1, 0] - R[0, 1]) / 2.0])
    return angle / (2.0 * np.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


@dataclass(frozen=True)
class Z1Joint:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    effort: float


class Z1Kinematics:
    """Serial FK/Jacobian and bounded damped-least-squares IK.

    The model follows URDF semantics exactly: each joint origin is applied in
    its parent link frame, then the joint rotation is applied around the
    declared axis. Fixed links (including the official Unitree gripper) are
    included in the chain but do not consume a q entry.
    """

    def __init__(self, joints: list[Z1Joint], tip_link: str, *, root_link: str):
        self.joints = tuple(joints)
        self.tip_link = tip_link
        self.root_link = root_link
        self.arm_joints = tuple(j for j in self.joints if j.joint_type != "fixed")
        self.joint_names = tuple(j.name for j in self.arm_joints)
        self.limits = np.asarray([(j.lower, j.upper) for j in self.arm_joints], dtype=float)
        self.efforts = np.asarray([j.effort for j in self.arm_joints], dtype=float)

    @classmethod
    def from_urdf(cls, path: str | Path, *, tip_link: str | None = None) -> "Z1Kinematics":
        root = ET.parse(path).getroot()
        all_joints = []
        for elem in root.findall("joint"):
            parent, child = elem.find("parent"), elem.find("child")
            if parent is None or child is None:
                continue
            limit = elem.find("limit")
            jt = elem.attrib["type"]
            lower = float(limit.attrib.get("lower", "0")) if limit is not None else 0.0
            upper = float(limit.attrib.get("upper", "0")) if limit is not None else 0.0
            effort = float(limit.attrib.get("effort", "nan")) if limit is not None else float("nan")
            axis_elem = elem.find("axis")
            axis_text = axis_elem.attrib.get("xyz", "0 0 0") if axis_elem is not None else "0 0 0"
            all_joints.append(Z1Joint(elem.attrib["name"], parent.attrib["link"], child.attrib["link"], jt, _transform(_xyz(elem.find("origin")), _rpy(elem.find("origin"))), np.asarray([float(v) for v in axis_text.split()], dtype=float), lower, upper, effort))
        by_parent = {}
        for joint in all_joints:
            by_parent.setdefault(joint.parent, []).append(joint)
        roots = {j.parent for j in all_joints} - {j.child for j in all_joints}
        root_link = "link00" if "link00" in roots else sorted(roots)[0]
        requested_tip = tip_link or ("gripperStator" if any(j.child == "gripperStator" for j in all_joints) else "link06")
        path_joints = []
        current = root_link
        while current != requested_tip:
            candidates = by_parent.get(current, [])
            if len(candidates) != 1:
                raise ValueError(f"URDF chain from {root_link} to {requested_tip} is ambiguous at {current}: {candidates}")
            joint = candidates[0]
            path_joints.append(joint)
            current = joint.child
        if not path_joints or current != requested_tip:
            raise ValueError(f"tip link {requested_tip!r} not reachable from {root_link!r}")
        return cls(path_joints, requested_tip, root_link=root_link)

    def fk(self, q: np.ndarray | list[float], *, include_gripper: bool = True) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.arm_joints),):
            raise ValueError(f"expected {len(self.arm_joints)} joint values, got {q.shape}")
        T = np.eye(4)
        qi = 0
        for joint in self.joints:
            if joint.joint_type == "fixed":
                T = T @ joint.origin
                continue
            if not include_gripper and joint.name == "jointGripper":
                break
            T = T @ joint.origin
            T[:3, :3] = T[:3, :3] @ _axis_angle(joint.axis, q[qi])
            qi += 1
        return T

    def fk_with_jacobian(self, q: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.arm_joints),):
            raise ValueError(f"expected {len(self.arm_joints)} joint values, got {q.shape}")
        T = np.eye(4)
        origins, axes = [], []
        qi = 0
        for joint in self.joints:
            T = T @ joint.origin
            if joint.joint_type == "fixed":
                continue
            origins.append(T[:3, 3].copy())
            axes.append(T[:3, :3] @ (joint.axis / max(np.linalg.norm(joint.axis), 1e-12)))
            T[:3, :3] = T[:3, :3] @ _axis_angle(joint.axis, q[qi])
            qi += 1
        p = T[:3, 3]
        J = np.zeros((6, len(origins)))
        for i, (origin, axis) in enumerate(zip(origins, axes)):
            J[:3, i] = np.cross(axis, p - origin)
            J[3:, i] = axis
        return T, J

    def inverse_kinematics(self, target: np.ndarray, initial: np.ndarray | list[float] | None = None, *, max_iters: int = 300, position_tolerance: float = 2e-4, orientation_tolerance: float = 2e-3, damping: float = 2e-3, step_limit: float = 0.12) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        target = np.asarray(target, dtype=float)
        if target.shape != (4, 4):
            raise ValueError("target must be a 4x4 homogeneous transform")
        q = np.asarray(initial if initial is not None else np.mean(self.limits, axis=1), dtype=float).copy()
        if q.shape != (len(self.arm_joints),):
            raise ValueError(f"expected {len(self.arm_joints)} initial joint values")
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        for iteration in range(max_iters):
            actual, J = self.fk_with_jacobian(q)
            error = np.r_[target[:3, 3] - actual[:3, 3], _rotvec(target[:3, :3] @ actual[:3, :3].T)]
            pos_err, ori_err = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
            if pos_err <= position_tolerance and ori_err <= orientation_tolerance:
                return q, {"success": True, "iterations": iteration, "position_error_m": pos_err, "orientation_error_rad": ori_err}
            update = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(6), error)
            update = np.clip(update, -step_limit, step_limit)
            q = np.clip(q + update, self.limits[:, 0], self.limits[:, 1])
        actual = self.fk(q)
        return q, {"success": False, "iterations": max_iters, "position_error_m": float(np.linalg.norm(target[:3, 3] - actual[:3, 3])), "orientation_error_rad": float(np.linalg.norm(_rotvec(target[:3, :3] @ actual[:3, :3].T)))}

