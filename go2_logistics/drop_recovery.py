"""Bounded recovery using fresh RGB-D and the original parcel identity.

The camera is a repositionable external inspection camera, not an onboard
sensor. Its search anchor comes from the last gripper pose, never parcel truth.
Initial scope is a single upright parcel on an unobstructed floor patch.
"""
from pathlib import Path
import copy

from .interfaces import Pose2D


def configure_recovery_camera(cfg):
    camera = cfg.scene.parcel_camera.copy()
    camera.prim_path = '{ENV_REGEX_NS}/RecoveryCamera'
    # Moving the camera prim does not refresh CameraData extrinsics by default.
    # RGB-D backprojection must use the pose of this observation, not startup.
    camera.update_latest_camera_pose = True
    cfg.scene.recovery_camera = camera


class DroppedParcelRecovery:
    def __init__(self, env, arm, output_dir, inject=False, camera_key='recovery_camera', clock=None, right_fallen=False):
        from .camera_perception import RedParcelPerception
        self.env = env.unwrapped
        self.arm = arm
        self.output_dir = Path(output_dir)
        self.perception = RedParcelPerception(env, arm, self.output_dir, decode_code=True)
        self.perception.camera = self.env.scene[camera_key]
        self.perception.estimate_geometry = getattr(arm,'visual_geometry',False)
        self.arm.contact_loss_detection = True
        self.inject_pending = inject
        self.events = []
        self.clock = clock
        self.right_fallen=right_fallen
        self.bridge = None
        self.navigate = None

    def bind(self, bridge, navigate):
        self.bridge, self.navigate = bridge, navigate

    def _record(self, event):
        if self.clock is not None:
            event['tick'] = self.clock()
        self.events.append(event)

    def _clear_view(self, object_id, phase):
        tool, orientation = self.arm._tool_pose()
        target = tool.clone()
        target[2] = max(float(tool[2]), .45)
        cleared = (float(target[2]-tool[2]) < .01 or
                   self.arm._move_tool(target, orientation.clone(), phase))
        self._record(dict(event='view_clearance', object_id=object_id, phase=phase,
                          success=bool(cleared),
                          tool_position=self.arm._tool_pose()[0].cpu().tolist()))
        if not cleared:
            self.arm.last_error = 'Empty gripper could not clear the drop search view'
        return bool(cleared)

    def _grasp_feasible(self, object_id):
        """Recheck the latest observation after every motion, never stale size."""
        center=self.arm.detected_position
        geometry=self.arm.detected_geometry
        half_height=geometry['height_m']/2 if geometry is not None else .04
        if center is None or abs(center[2]-half_height) > .015:
            self.arm.last_error = f'Dropped parcel is not on floor: visual center={center}'
            return False
        if geometry is not None and geometry['grasp_width_m']>2*self.arm.config.finger_open+.014:
            self.arm.last_error=('Observed fallen parcel is identified but exceeds the current gripper aperture; '
                                 'contact righting or a reachable side grasp is required')
            self._record(dict(event='grasp_feasibility',object_id=object_id,success=False,
                reason='observed_width_exceeds_aperture',observed_width_m=geometry['grasp_width_m'],
                finger_open_m=self.arm.config.finger_open))
            return False
        return True

    def _active_observation(self, object_id, anchor, attempt, suffix=''):
        """Fuse a horizontal face with a fresh, spatially associated side QR.

        This is a bounded external-camera inspection, not a box-state lookup.
        An identity read alone never supplies a grasp pose.
        """
        import torch
        from .recovery_views import side_view_poses,label_on_observed_box,geometry_view_pose
        camera=self.perception.camera
        original_geometry=self.perception.estimate_geometry
        self.perception.estimate_geometry=True
        try:
            hint=(self.perception.last_detection or {}).get('center_world')
            eye,target=geometry_view_pose(anchor,hint)
            eye=torch.tensor(eye,device=self.env.device)
            target=torch.tensor(target,device=self.env.device)
            camera.set_world_poses_from_view(eye.unsqueeze(0),target.unsqueeze(0))
            self.perception.output_dir=self.output_dir/f'recovery_{attempt:02d}{suffix}_geometry'
            if self.perception.locate(object_id):
                self._record(dict(event='active_geometry',object_id=object_id,success=True,
                    geometry=copy.deepcopy(self.arm.detected_geometry),identity_source='same_top_view'))
                return True
            if not self.perception.locate_geometry():
                self._record(dict(event='active_geometry',object_id=object_id,success=False))
                return False
            geometry=copy.deepcopy(self.arm.detected_geometry)
            detection=copy.deepcopy(self.perception.last_detection)
            center=geometry['center_world']
            self._record(dict(event='active_geometry',object_id=object_id,success=True,geometry=geometry))
            for index,(eye,target) in enumerate(side_view_poses(center)):
                camera.set_world_poses_from_view(torch.tensor([eye],device=self.env.device),
                                                torch.tensor([target],device=self.env.device))
                self.perception.output_dir=self.output_dir/f'recovery_{attempt:02d}{suffix}_side_{index:02d}'
                label=self.perception.inspect_label()
                associated=bool(label and label['parcel_code']==object_id and label_on_observed_box(label,geometry))
                self._record(dict(event='side_identity_view',object_id=object_id,view=index,
                                  associated=associated,label=label))
                if associated:
                    # The side scan did not advance physics. Reuse only this
                    # same-time depth fit and explicitly record both sources.
                    detection.update(object_id=object_id,parcel_code=label['parcel_code'],
                        identity_source='fresh_side_rgb_depth_associated_with_top_geometry',
                        side_label=label,geometry=geometry)
                    self.perception.last_detection=detection
                    self.arm.detected_position=tuple(center)
                    self.arm.detected_geometry=geometry
                    return True
            self.arm.detected_position=None
            self.arm.detected_geometry=None
            return False
        finally:
            self.perception.estimate_geometry=original_geometry

    def inject_after_transit(self):
        if not self.inject_pending or not self.arm.held:
            return
        self.inject_pending = False
        self.arm._set_phase('drop_test_stop')
        for _ in range(60):
            self.arm._step()
        tool, orientation = self.arm._tool_pose()
        target = tool.clone()
        target[2] = .172
        if not self.arm._move_tool(target, orientation.clone(), 'drop_test_lowering'):
            raise RuntimeError('Could not reach controlled drop height')
        self.arm._set_phase('drop_test_opening')
        self._record(dict(event='injection', method='open_fingers_only',
                                tool_position=self.arm._tool_pose()[0].cpu().tolist()))
        print(f'DROP_RECOVERY {self.events[-1]}', flush=True)
        self.arm._finger_target = self.arm.config.finger_open
        # The contact monitor must raise PayloadLost; injection itself does not
        # declare loss, alter held, or write the rigid body's pose/velocity.
        for _ in range(100):
            self.arm._step()
        raise RuntimeError('Drop injection did not produce a detected contact loss')

    def __call__(self, object_id):
        import torch
        from dataclasses import replace
        self.arm._finger_target = self.arm.config.finger_open
        self.arm._set_phase('drop_settling')
        for _ in range(80):
            self.arm._step()
        anchor = self.arm.loss_events[-1]['tool_position']
        # An opened gripper can still cover the top label. Clear the observed
        # drop area by moving the empty tool vertically before changing views.
        # The target uses only the tool pose, never the parcel's truth pose.
        if not self._clear_view(object_id,'drop_clear_view'):
            return False
        attempt = sum(event['event'] == 'search' for event in self.events) + 1
        self.arm._set_phase('drop_visual_search')
        camera = self.perception.camera
        # Small bounded local scan about the last gripper position. No truth
        # position is used to point the camera or select the docking goal.
        found = False
        for index, (dx, dy) in enumerate(((0.,0.),(.12,0.),(-.12,0.),(0.,.12),(0.,-.12))):
            target = torch.tensor((anchor[0]+dx, anchor[1]+dy, .04),device=self.env.device)
            eye = target + torch.tensor((.35,0.,.76),device=self.env.device)
            camera.set_world_poses_from_view(eye.unsqueeze(0),target.unsqueeze(0))
            self.perception.output_dir = self.output_dir / f'recovery_{attempt:02d}_view_{index:02d}'
            if self.perception.locate(object_id):
                found = True
                break
        active_view=False
        if not found:
            active_view=self._active_observation(object_id,anchor,attempt)
            found=active_view
        self._record(dict(event='search', object_id=object_id, found=found,
                                anchor=anchor, detection=self.perception.last_detection,
                                source='rendered_rgb_depth_external_camera'))
        print(f'DROP_RECOVERY {self.events[-1]}',flush=True)
        if not found:
            self.arm.last_error = 'Dropped parcel identity not visible in bounded RGB-D search'
            return False
        center = self.arm.detected_position
        # Existing perception fits an upright box of known height; fail closed
        # instead of attempting to lift a fallen, wide face with narrow fingers.
        half_height = self.arm.detected_geometry['height_m']/2 if self.arm.detected_geometry is not None else .04
        if abs(center[2]-half_height) > .015:
            self.arm.last_error = f'Dropped parcel is not upright on floor: visual center={center}'
            return False
        geometry=self.arm.detected_geometry
        if geometry is not None and geometry['grasp_width_m']>2*self.arm.config.finger_open+.014:
            if self.right_fallen and active_view:
                from .contact_righting import execute_contact_righting
                before=copy.deepcopy(self.perception.last_detection)
                if not execute_contact_righting(self.arm,before,self._record):
                    self.arm.last_error='Bounded contact-righting reference did not complete'
                    return False
                self.arm._finger_target=self.arm.config.finger_open
                for _ in range(80):self.arm._step()
                if not self._clear_view(object_id,'righting_clear_view'):
                    return False
                # Never infer an upright box from the push command. Search
                # again for the original ID and freshly fit its actual face.
                if not self._active_observation(object_id,center,attempt,'_after_righting'):
                    self.arm.last_error='Original parcel could not be reobserved after contact righting'
                    return False
                geometry=self.arm.detected_geometry
                center=self.arm.detected_position
                half_height=geometry['height_m']/2
                self._record(dict(event='righting_reobservation',object_id=object_id,
                    detection=copy.deepcopy(self.perception.last_detection)))
        if not self._grasp_feasible(object_id):
            return False
        self.arm._set_phase('drop_redock')
        if not self.navigate(Pose2D(center[0]-.58,center[1])):
            self.arm.last_error = 'Could not redock at the observed dropped parcel'
            return False
        # Acquire another observation after base motion, using the same camera
        # pose. Base movement can push objects; stale grasp targets are unsafe.
        self.perception.output_dir = self.output_dir / f'recovery_{attempt:02d}_redock'
        reobserved=(self._active_observation(object_id,center,attempt,'_redock') if active_view
                    else self.perception.locate(object_id))
        if not reobserved:
            self.arm.last_error = 'Dropped parcel was lost from view after redocking'
            return False
        if not self._grasp_feasible(object_id):
            return False
        half_height=self.arm.detected_geometry['height_m']/2 if self.arm.detected_geometry is not None else .04
        original = self.arm.config
        destination = (*original.dropoff_position[:2],.42+half_height)
        self.arm.config = replace(original,reset_before_grasp=True,dropoff_position=destination)
        try:
            ok = self.arm.grasp(object_id)
        finally:
            self.arm.config = replace(original,dropoff_position=destination)
        self._record(dict(event='regrasp', object_id=object_id, success=ok,
                                detected_position=self.arm.detected_position))
        print(f'DROP_RECOVERY {self.events[-1]}',flush=True)
        return ok
