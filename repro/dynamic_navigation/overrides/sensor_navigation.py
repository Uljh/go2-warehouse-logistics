"""Online range mapping and A* navigation; no obstacle poses enter the planner.

Localization is simulator ground truth. The idealized two-layer 360 degree
scanner uses PhysX scene queries, excluding only robot/payload self geometry.
Dynamic mode clears occupied cells only along subsequently observed free rays.
"""
import heapq
import itertools
import json
import math
from pathlib import Path


def bounded_recovery_velocity(velocity, current_pose, start_pose, max_displacement=.15):
    """Return a body-frame recovery command, zeroed past the travel cap."""
    if velocity is None:
        return None
    if math.dist((current_pose.x,current_pose.y),(start_pose.x,start_pose.y))>=max_displacement-1e-9:
        return (0.,0.,0.)
    return tuple(float(value) for value in velocity)


def recovery_forward_is_clear(returns, pose, travel_m=.15, drift_margin_m=.10):
    """Check the swept body/arm footprint in both measured range layers.

    Use the navigation footprints (low radius .35; high x [-.45,.85],
    half-width .40), rotated into the current body frame. Include the full
    experimental travel and a drift allowance rather than only center range.
    This remains a two-layer ideal sensor check, not a full 3-D certificate.
    """
    cos_yaw, sin_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
    for item in returns:
        if len(item) < 3 or not all(math.isfinite(float(v)) for v in item[:3]):
            raise ValueError('Invalid recovery range return')
        if item[2] not in (0, 1):
            raise ValueError('Unknown recovery range layer')
        dx, dy = float(item[0]) - pose.x, float(item[1]) - pose.y
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        if item[2] == 0:
            closest_x = min(travel_m, max(0., forward))
            blocked = math.hypot(forward-closest_x, lateral) <= .35+drift_margin_m
        else:
            blocked = (-.45-drift_margin_m <= forward <= .85+travel_m+drift_margin_m
                       and abs(lateral) <= .40+drift_margin_m)
        if blocked:
            return False
    return True


class OccupancyGrid:
    def __init__(self, resolution=.10, bounds=(-8., 8., -2., 11.)):
        self.resolution = resolution
        self.bounds = bounds
        self.blocked = set()
        self.observed = set()
        self.risk = {}
        # Coordination is a separate planning layer. Free sensor rays must
        # never erase the centralized peer clearance used by the traffic guard.
        self.coordination_keepouts = []
        self.coordination_blocked_checks = 0
        self.coordination_plan_interventions = 0

    def set_coordination_keepouts(self, positions, radius=1.50):
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('Peer coordination radius must be positive and finite')
        circles = []
        for point in positions:
            if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
                raise ValueError('Peer coordination positions must be finite XY pairs')
            circles.append((float(point[0]), float(point[1]), float(radius)))
        self.coordination_keepouts = circles

    def coordination_clear_line(self, start, end):
        """Test the entire segment against peer discs, including between cells."""
        dx, dy = end[0]-start[0], end[1]-start[1]
        length_squared = dx*dx+dy*dy
        for x, y, radius in self.coordination_keepouts:
            fraction = max(0., min(1., ((x-start[0])*dx+(y-start[1])*dy)/length_squared)) if length_squared else 0.
            closest = (start[0]+fraction*dx, start[1]+fraction*dy)
            if math.dist(closest, (x,y)) <= radius:
                self.coordination_blocked_checks += 1
                return False
        return True

    def cell(self, point):
        return tuple(round(v / self.resolution) for v in point[:2])

    def point(self, cell):
        return tuple(v * self.resolution for v in cell)

    def valid(self, cell):
        x, y = self.point(cell)
        a, b, c, d = self.bounds
        return (a <= x <= b and c <= y <= d and cell not in self.blocked
                and self.coordination_clear_line((x,y), (x,y)))

    def add_hit(self, point, layer=0):
        cell = self.cell(point)
        key = (*cell, layer)
        if key in self.observed:
            return
        self.observed.add(key)
        # Low returns protect the trunk/legs. High returns also protect the
        # forward-reaching arm and parcel. Heading stays approximately zero.
        for ix in range(-9, 6):
            for iy in range(-5, 6):
                dx, dy = ix*self.resolution, iy*self.resolution
                inside = math.hypot(dx, dy) <= .35 if layer == 0 else (-.85 <= dx <= .45 and abs(dy) <= .40)
                if inside:
                    occupied = (cell[0]+ix, cell[1]+iy)
                    if occupied not in self.blocked:
                        self.blocked.add(occupied)
                        for rx in range(-3,4):
                            for ry in range(-3,4):
                                near = (occupied[0]+rx,occupied[1]+ry)
                                self.risk[near] = max(self.risk.get(near,0),4-max(abs(rx),abs(ry)))

    def clear_line(self, start, end, prefer_clearance=False):
        if not self.coordination_clear_line(start, end):
            return False
        distance = math.dist(start, end)
        count = max(1, math.ceil(distance / (self.resolution*.25)))
        previous = self.cell(start)
        allowed_risk = max(self.risk.get(previous,0),self.risk.get(self.cell(end),0))
        for i in range(count+1):
            cell = self.cell((start[0]+(end[0]-start[0])*i/count,start[1]+(end[1]-start[1])*i/count))
            if not self.valid(cell):
                return False
            if prefer_clearance and self.risk.get(cell,0)>allowed_risk:
                return False
            if cell[0] != previous[0] and cell[1] != previous[1]:
                if not self.valid((cell[0],previous[1])) or not self.valid((previous[0],cell[1])):
                    return False
            previous = cell
        return True

    def update_scan(self, returns, free_segments):
        """Clear only directly observed free ray segments, never aged/hidden hits."""
        free=set()
        for start,end,layer in free_segments:
            distance=math.dist(start,end)
            # Leave a terminal margin for discretization and grazing surfaces.
            count=max(1,math.ceil(distance/(self.resolution*.5)))
            for index in range(max(0,count-4)):
                fraction=index/count
                cell=self.cell((start[0]+(end[0]-start[0])*fraction,start[1]+(end[1]-start[1])*fraction))
                free.add((*cell,layer))
        incoming={(*self.cell((x,y)),layer) for x,y,layer,_ in returns}
        removed=(self.observed & free)-incoming
        if removed:
            remaining=self.observed-removed
            self.observed=set();self.blocked=set();self.risk={}
            for x,y,layer in remaining:
                self.add_hit(self.point((x,y)),layer)
        for x,y,layer,_ in returns:
            self.add_hit((x,y),layer)
        return len(removed)

    def plan(self, start, goal):
        """8-connected A*, no corner cutting; simplify only visible segments."""
        if not self.coordination_clear_line(start,start) or not self.coordination_clear_line(goal,goal):
            raise RuntimeError('Navigation endpoint inside peer coordination margin')
        if not self.coordination_clear_line(start, goal):
            self.coordination_plan_interventions += 1
        first, last = self.cell(start), self.cell(goal)
        if not self.valid(first) or not self.valid(last):
            raise RuntimeError(f"Navigation pose inside sensed safety margin: start={start}, goal={goal}")
        if self.clear_line(start, goal, prefer_clearance=True):
            return [tuple(goal)]
        serial = itertools.count()
        queue = [(0., next(serial), first)]
        cost, parent = {first: 0.}, {}
        directions = [(x,y) for x in (-1,0,1) for y in (-1,0,1) if x or y]
        while queue:
            _, _, current = heapq.heappop(queue)
            if current == last:
                cells = [last]
                while cells[-1] != first:
                    cells.append(parent[cells[-1]])
                points = [tuple(start)] + [self.point(c) for c in reversed(cells[:-1])] + [tuple(goal)]
                simplified, index = [], 0
                while index < len(points)-1:
                    target = len(points)-1
                    while target > index+1 and not self.clear_line(points[index], points[target], prefer_clearance=True):
                        target -= 1
                    simplified.append(points[target])
                    index = target
                if not all(self.clear_line(a,b) for a,b in zip([tuple(start)]+simplified,simplified)):
                    raise RuntimeError('Navigation path failed continuous clearance validation')
                return simplified
            for dx, dy in directions:
                neighbor = (current[0]+dx, current[1]+dy)
                if not self.valid(neighbor):
                    continue
                if not self.coordination_clear_line(self.point(current), self.point(neighbor)):
                    continue
                if dx and dy and (not self.valid((current[0]+dx,current[1])) or not self.valid((current[0],current[1]+dy))):
                    continue
                candidate = cost[current]+math.hypot(dx,dy)*(1+2*self.risk.get(neighbor,0))
                if candidate < cost.get(neighbor, float('inf')):
                    cost[neighbor], parent[neighbor] = candidate, current
                    heapq.heappush(queue, (candidate+math.dist(neighbor,last),next(serial),neighbor))
        raise RuntimeError("No collision-free route in sensed occupancy map")

    def split_axis_segments(self, start, path):
        """Replace diagonals only when both orthogonal legs retain clearance.

        This is an optional locomotion adaptation, not a relaxation of the
        occupancy footprint. If neither corner is safe, preserve the original
        already checked segment. Prefer the longer first leg to avoid tiny
        corrections before the main traverse.
        """
        result = []
        previous = tuple(start)
        split_count = 0
        for target in path:
            target = tuple(target)
            dx, dy = target[0]-previous[0], target[1]-previous[1]
            if min(abs(dx), abs(dy)) > .20:
                corners = [(target[0], previous[1]), (previous[0], target[1])]
                if abs(dy) > abs(dx):
                    corners.reverse()
                for corner in corners:
                    if (self.clear_line(previous, corner, prefer_clearance=True)
                            and self.clear_line(corner, target, prefer_clearance=True)):
                        result.append(corner)
                        split_count += 1
                        break
            result.append(target)
            previous = target
        return result, split_count


class PhysXRangeScanner:
    def __init__(self, env, robot_key='robot', self_path=None, payload_path=None,
                 range_profile='ideal', seed=0, measurement_log=None, pose_observation=None,audit_paths=False):
        from omni.physx import get_physx_scene_query_interface
        self.env = env.unwrapped
        self.robot_key = robot_key
        self.self_path = self_path
        self.payload_path = payload_path
        self.query = get_physx_scene_query_interface()
        if range_profile not in ('ideal', 'perturbed'):
            raise ValueError('Unknown range profile')
        from .range_measurement import RangeMeasurementModel
        self.measurement_model = RangeMeasurementModel(seed) if range_profile == 'perturbed' else None
        self.measurement_log = Path(measurement_log) if measurement_log else None
        self.scan_number = 0
        self.last_hit_paths = {}
        self.audit_paths = bool(audit_paths)
        if pose_observation is not None and not callable(pose_observation):
            raise ValueError('Pose observation provider must be callable')
        self.pose_observation = pose_observation
        self.directions = [(math.cos(math.radians(a)),math.sin(math.radians(a)),0.) for a in range(0,360,2)]

    def scan(self):
        robot = self.env.scene[self.robot_key]
        root = getattr(robot.data.root_pos_w,'torch',robot.data.root_pos_w)[0].cpu().tolist()
        origin = getattr(self.env.scene.env_origins,'torch',self.env.scene.env_origins)[0].cpu().tolist()
        observation = self.pose_observation() if getattr(self, 'pose_observation', None) is not None else None
        true_origin = (root[0]-origin[0],root[1]-origin[1])
        if observation is not None and math.dist(true_origin, (observation.truth.x, observation.truth.y)) > 1e-6:
            raise RuntimeError('Range scan localization is stale relative to physics')
        if observation is not None:
            quat = getattr(robot.data.root_quat_w,'torch',robot.data.root_quat_w)[0].cpu().tolist()
            x,y,z,w = quat
            yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            yaw_error = (yaw-observation.truth.yaw+math.pi)%(2*math.pi)-math.pi
            if abs(yaw_error) > 1e-6:
                raise RuntimeError('Range scan localization heading is stale relative to physics')
        map_origin = (observation.estimate.x, observation.estimate.y) if observation is not None else true_origin
        returns = []
        self.free_segments=[]
        beam_records=[]
        self.scan_number += 1
        self.last_hit_paths = {}
        for layer, offset in enumerate((-.09, .29)):
            eye = (root[0],root[1],root[2]+offset)
            for direction in self.directions:
                hits = []
                def report(hit):
                    def field(name):
                        return hit.get(name) if isinstance(hit,dict) else getattr(hit,name)
                    path = str(field('rigid_body'))
                    if self.self_path is None:
                        ignored = '/Robot/' in path or path.endswith('/Robot') or '/Parcel' in path
                    else:
                        ignored = path == self.self_path or path.startswith(self.self_path+'/')
                        if self.payload_path:
                            ignored = ignored or path == self.payload_path or path.startswith(self.payload_path+'/')
                    if not ignored:
                        pos = field('position')
                        hits.append((float(field('distance')), (float(pos[0])-origin[0],float(pos[1])-origin[1]),path))
                    return True
                self.query.raycast_all(eye,direction,5.,report)
                nearest_path=min(hits)[2] if hits else None
                if nearest_path is not None:
                    self.last_hit_paths[nearest_path]=self.last_hit_paths.get(nearest_path,0)+1
                if self.measurement_model is not None or observation is not None:
                    from .range_measurement import ray_observation
                    raw_distance = min(hits)[0] if hits else None
                    measured = (self.measurement_model.measure(raw_distance) if self.measurement_model is not None
                        else dict(status='hit' if raw_distance is not None else 'no_return',
                                  distance_m=raw_distance if raw_distance is not None else 5.))
                    start, map_direction = (observation.ray_in_map(direction) if observation is not None
                        else (true_origin, direction))
                    hit, free = ray_observation(start, map_direction, layer, measured)
                    if hit is not None:
                        returns.append(hit)
                    if free is not None:
                        self.free_segments.append(free)
                    beam_records.append(dict(layer=layer, direction=map_direction,
                        physical_direction_world=direction, raw_distance_m=raw_distance,
                        physical_hit_prim_path=nearest_path,
                        measurement=measured, occupied_return=hit, free_segment=free))
                    continue
                if hits:
                    distance, point, _ = min(hits)
                    returns.append((point[0],point[1],layer,distance))
                else:
                    point=(eye[0]-origin[0]+direction[0]*5.,eye[1]-origin[1]+direction[1]*5.)
                self.free_segments.append(((eye[0]-origin[0],eye[1]-origin[1]),point,layer))
                if getattr(self,'audit_paths',False):
                    beam_records.append(dict(layer=layer,physical_direction_world=direction,
                        physical_hit_prim_path=nearest_path,raw_distance_m=min(hits)[0] if hits else None,
                        occupied_return=returns[-1] if hits else None,free_segment=self.free_segments[-1]))
        if (self.measurement_model is not None or observation is not None or getattr(self,'audit_paths',False)) and self.measurement_log is not None:
            self.measurement_log.parent.mkdir(parents=True, exist_ok=True)
            with self.measurement_log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(scan=self.scan_number, origin_xy=map_origin,
                    true_origin_xy=true_origin, pose_observation=observation.record() if observation is not None else None,
                    beams=beam_records))+'\n')
        if self.measurement_model is not None:
            # Do not let an outage turn an unobserved corridor into a clear path.
            # Current response is task failure/zero-command cleanup, not recovery.
            coverage = [sum(b['layer']==layer and b['measurement']['status'] in ('hit','no_return')
                            for b in beam_records)/len(self.directions) for layer in (0, 1)]
            if min(coverage) < .8:
                raise RuntimeError(f'Range coverage below 80 percent: {coverage}')
        return returns


class SensorNavigator:
    def __init__(self, env, output_dir=None, *, dynamic_map=False, minimum_forward_command=.30,
                 stalled_forward_command=None, minimum_positive_forward_command=None,
                 stalled_lateral_command=None, axis_routes=False, travel_heading=False,
                 loaded_recovery_trigger_height=.285, peer_positions=None, control_period_s=.02,
                 vector_tracking=False, docking_axis_scope=None,dynamic_yield=False):
        if not math.isfinite(control_period_s) or not 0<control_period_s<=.02:
            raise ValueError('Navigation control period must be in (0, 20 ms]')
        self.control_period_s=control_period_s
        from .dynamic_yield import RangeYield
        self.range_yield=RangeYield() if dynamic_yield else None
        self.range_yield_telemetry=None
        self.range_yield_steps=0
        self.vector_tracking=bool(vector_tracking)
        if docking_axis_scope is not None and (not callable(docking_axis_scope) or not self.vector_tracking):
            raise ValueError('Docking axis persistence requires vector tracking and a scope callback')
        from .docking_axis import DockingAxisHold
        self.docking_axis_scope=docking_axis_scope
        self.docking_axis=DockingAxisHold() if docking_axis_scope is not None else None
        self.docking_axis_telemetry=None
        if not math.isfinite(loaded_recovery_trigger_height) or not .285<=loaded_recovery_trigger_height<=.305:
            raise ValueError('Loaded recovery trigger height must be finite in [.285,.305]')
        self.loaded_recovery_trigger_height=loaded_recovery_trigger_height
        self.scanner = PhysXRangeScanner(env)
        self.grid = OccupancyGrid()
        if peer_positions is not None and not callable(peer_positions):
            raise ValueError('Peer positions must be a callable or None')
        self.peer_positions = peer_positions
        self.coordination_refreshes = 0
        self.coordination_max_peers = 0
        self.output_dir = Path(output_dir) if output_dir else None
        self.events = []
        self.scans = []
        self.plans = 0
        self.detours = 0
        self.stops = 0
        self.travel = 0.
        self.dynamic_map=dynamic_map
        self.minimum_forward_command=minimum_forward_command
        if minimum_positive_forward_command is not None and (not math.isfinite(minimum_positive_forward_command) or not .30<=minimum_positive_forward_command<=.50):
            raise ValueError('Positive forward floor must be finite in [.30,.50]')
        self.minimum_positive_forward_command=minimum_positive_forward_command
        self.stalled_forward_command=stalled_forward_command
        if stalled_lateral_command is not None and (not math.isfinite(stalled_lateral_command)
                or not .22<=stalled_lateral_command<=.35):
            raise ValueError('Lateral stall command must be finite in [.22,.35]')
        self.stalled_lateral_command=stalled_lateral_command
        self.axis_routes=axis_routes
        self.travel_heading=travel_heading
        self.cruise_events=[]
        self.axis_route_splits=0
        self.lateral_stall_bursts=0
        self.motion_stall_bursts=0
        self.cleared_cells=0
        self.path_invalidations=0
        self.wait_steps=0
        self.stability_pauses=0
        self.recovering=False
        self.target_revision=0
        self.recovery_safety_stops=0
        self.recovery_sensor_failures=0
        self.recovery_payload_suppressed=0
        self.recovery_events=[]

    def _refresh_coordination_keepouts(self):
        if self.peer_positions is not None:
            self.grid.set_coordination_keepouts(self.peer_positions(), radius=1.50)
            self.coordination_refreshes += 1
            self.coordination_max_peers = max(self.coordination_max_peers, len(self.grid.coordination_keepouts))

    def recover_stance(self,bridge, *, settle_motion=False, minimum_duration_s=0.,
                       precision_manipulation=False):
        """Pause velocity tracking before low trunk clearance becomes a fall."""
        if (not math.isfinite(minimum_duration_s) or not 0. <= minimum_duration_s <= 3.
                or minimum_duration_s and not settle_motion):
            raise ValueError('Minimum braking duration requires motion settling and must be within 3 seconds')
        if precision_manipulation and not settle_motion:
            raise ValueError('Precision manipulation requires explicit motion settling')
        exit_height = .325 if precision_manipulation else .305
        exit_gravity = -.9995 if precision_manipulation else -.990
        linear_limit = .02 if precision_manipulation else .06
        angular_limit = .08 if precision_manipulation else .15
        settle_exit = settle_motion or bool(getattr(bridge, 'recovery_settle_motion', False))
        env=bridge.env.unwrapped
        data=env.scene['robot'].data
        def height():
            root=getattr(data.root_pos_w,'torch',data.root_pos_w)
            return float(root[0,2])-float(env.scene.env_origins[0,2])
        def gravity_z():
            gravity=getattr(data,'projected_gravity_b',None)
            return float(getattr(gravity,'torch',gravity)[0,2]) if gravity is not None else -1.
        def motion_settled():
            if not getattr(bridge,'recovery_step_ready',lambda:True)():
                return False
            if not settle_exit:
                return True
            linear=data.root_lin_vel_w
            angular=data.root_ang_vel_b
            linear=getattr(linear,'torch',linear)[0]
            angular=getattr(angular,'torch',angular)[0]
            return (math.sqrt(sum(float(x)**2 for x in linear))<linear_limit
                and math.sqrt(sum(float(x)**2 for x in angular))<angular_limit
                and (not precision_manipulation or abs(float(linear[2]))<.005))
        payload_attached=bool(getattr(bridge,'recovery_payload_held',lambda:False)())
        trigger_height=self.loaded_recovery_trigger_height if payload_attached else .285
        if (minimum_duration_s == 0. and height()>= (exit_height if settle_motion else trigger_height)
                and gravity_z()<=(exit_gravity if settle_motion else -.985)
                and (motion_settled() if settle_motion else True)):
            return
        self.stability_pauses+=1
        print(f'SENSOR_STANCE_PAUSE height={height():.3f} gravity_z={gravity_z():.3f}',flush=True)
        stable=0
        self.recovering=True
        recovery_velocity_body=getattr(bridge,'recovery_velocity_body',None)
        if recovery_velocity_body and payload_attached:
            # The loaded forward-recovery trial failed. Keep it disabled for
            # held parcels; scanner self-exclusion is not a grasp-state signal.
            self.recovery_payload_suppressed+=1
            recovery_velocity_body=None
        recovery_start=bridge.pose() if recovery_velocity_body else None
        event=dict(start_height_m=height(),start_gravity_z=gravity_z(),payload_held=payload_attached,
                   trigger_height_m=trigger_height,
                   require_settled_exit=settle_exit,
                   minimum_duration_s=minimum_duration_s,
                   precision_manipulation=bool(precision_manipulation),
                   exit_thresholds=dict(height_m=exit_height,gravity_z=exit_gravity,
                       linear_speed_mps=linear_limit,angular_speed_radps=angular_limit,
                       vertical_speed_mps=.005 if precision_manipulation else None),
                   reason='transition_motion_settle' if settle_motion else 'low_clearance_or_tilt',
                   configured_velocity_body=getattr(bridge,'recovery_velocity_body',None),
                   requested_forward_steps=0,control_steps=0,
                   max_displacement_m=0. if recovery_start is not None else None,resumed=False,
                   safety_stop=False,sensor_failed=False,travel_cap_reached=False,
                   end_height_m=None,end_gravity_z=None)
        self.recovery_events.append(event)
        recovery_clear=False
        try:
            for step in range(self._ticks(150)):
                recovery_velocity=bounded_recovery_velocity(
                    recovery_velocity_body,bridge.pose(),recovery_start) if recovery_start is not None else None
                # Refresh at the same 10 Hz rate as normal navigation. A
                # blocked/failed scan latches zero for the rest of this event.
                if recovery_start is not None and (step==0 or step%self._ticks(5)==0 and recovery_clear):
                    try:
                        if self.scanner is None:
                            raise RuntimeError('Recovery range scanner unavailable')
                        pose=bridge.pose()
                        self._refresh_coordination_keepouts()
                        returns=self.scanner.scan()
                        self.scans.append(dict(pose=(pose.x,pose.y),returns=returns,purpose='recovery'))
                        recovery_clear=recovery_forward_is_clear(returns,pose)
                    except Exception as error:
                        recovery_clear=False
                        self.recovery_sensor_failures+=1
                        event['sensor_failed']=True
                        print(f'SENSOR_RECOVERY_SCAN_FAILED {error}',flush=True)
                    if not recovery_clear:
                        self.recovery_safety_stops+=1
                        event['safety_stop']=True
                        print('SENSOR_RECOVERY_FORWARD_BLOCKED swept_footprint_or_scan_unavailable',flush=True)
                if recovery_velocity is not None and not recovery_clear:
                    recovery_velocity=(0.,0.,0.)
                if recovery_start is not None and recovery_velocity==(0.,0.,0.) and recovery_clear:
                    event['travel_cap_reached']=True
                bridge.recovery_requested_velocity_body=recovery_velocity or (0.,0.,0.)
                event['requested_forward_steps']+=int(bridge.recovery_requested_velocity_body[0]>0.)
                event['control_steps']+=1
                if recovery_velocity is None:
                    bridge._zero_command()
                else:
                    bridge.set_body_velocity_command(*recovery_velocity)
                bridge._policy_step()
                if recovery_start is not None:
                    pose=bridge.pose()
                    event['max_displacement_m']=max(event['max_displacement_m'],
                        math.dist((pose.x,pose.y),(recovery_start.x,recovery_start.y)))
                stable=stable+1 if height()>exit_height and gravity_z()<exit_gravity and motion_settled() else 0
                if stable>=self._ticks(10) and (step+1)*self.control_period_s>=minimum_duration_s-1e-9:
                    event['resumed']=True
                    print(f'SENSOR_STANCE_RESUMED height={height():.3f}',flush=True)
                    return
            raise RuntimeError('Stance recovery could not restore trunk clearance and settled motion within 3 seconds '
                f'(height={height():.4f}m, gravity_z={gravity_z():.5f}, '
                f'settled={motion_settled()}, confirmation_frames={stable}/{self._ticks(10)})')
        finally:
            self.recovering=False
            bridge.recovery_requested_velocity_body=(0.,0.,0.)
            event.update(end_height_m=height(),end_gravity_z=gravity_z())
            event['final_exit_conditions']=dict(height_clear=height()>exit_height,
                gravity_clear=gravity_z()<exit_gravity,motion_or_step_ready=motion_settled(),
                confirmation_frames=stable,required_confirmation_frames=self._ticks(10))

    def _ticks(self, legacy_50hz_frames):
        """Preserve physical durations when the arm changes the shared loop rate."""
        return max(1,math.ceil(legacy_50hz_frames*.02/self.control_period_s-1e-9))

    def navigate(self, bridge, goal, transit=False, *, position_tolerance=None):
        from .interfaces import Pose2D
        self.docking_axis_telemetry=None
        if self.docking_axis is not None:
            self.docking_axis.reset()
        path = []
        target_xy = (goal.x,goal.y)
        tolerance = .20 if transit else bridge.config.position_tolerance
        if position_tolerance is not None:
            if not math.isfinite(position_tolerance) or not 0 < position_tolerance <= tolerance:
                raise ValueError('Station tolerance must tighten the existing arrival tolerance')
            tolerance = position_tolerance
        progress_origin = None
        progress_steps = 0
        burst_steps = 0
        lateral_burst_steps = 0
        cruise = None
        cruise_cooldown = 0
        latest_returns = []
        yield_command=None
        if self.range_yield is not None:
            from .dynamic_yield import RangeYield
            self.range_yield=RangeYield()
        for step in range(bridge.config.max_control_steps):
            self.docking_axis_telemetry=None
            if self.dynamic_map and hasattr(bridge,'env'):
                self.recover_stance(bridge)
            pose = bridge.pose()
            xy = (pose.x,pose.y)
            if cruise is None and math.dist(xy,target_xy) <= tolerance and abs(math.atan2(math.sin(pose.yaw-goal.yaw),math.cos(pose.yaw-goal.yaw))) <= bridge.config.yaw_tolerance:
                bridge._zero_command()
                return True
            if step % self._ticks(5) == 0:
                self._refresh_coordination_keepouts()
                returns = self.scanner.scan()
                latest_returns = returns
                self.scans.append(dict(pose=xy,returns=returns))
                if self.dynamic_map:
                    self.cleared_cells+=self.grid.update_scan(returns,self.scanner.free_segments)
                else:
                    for x,y,layer,_ in returns:
                        self.grid.add_hit((x,y),layer)
                if self.range_yield is not None:
                    was_active=bool(self.range_yield_telemetry and self.range_yield_telemetry.get('active'))
                    yield_command=self.range_yield.update(pose,returns,self._ticks(5)*self.control_period_s,
                        goal=target_xy,bounds=self.grid.bounds)
                    self.range_yield_telemetry=self.range_yield.telemetry
                    if yield_command is not None:
                        path=[];cruise=None
                        if not was_active:
                            print(f'SENSOR_DYNAMIC_YIELD {self.range_yield_telemetry}',flush=True)
                        command=bridge.env.unwrapped.command_manager.get_term('base_velocity').command
                        command[0,:]=command.new_tensor(yield_command)
                        self.range_yield_steps+=1
                        bridge._policy_step()
                        continue
                if cruise is not None:
                    cruise.check_scan(pose, returns, self.grid)
                elif not path or not all(self.grid.clear_line(a,b) for a,b in zip([xy]+path,path)):
                    if path:
                        self.path_invalidations+=1
                    bridge._zero_command()
                    # One measured zero-command step precedes each new plan.
                    bridge._policy_step()
                    self.stops += 1
                    pose = bridge.pose(); xy = (pose.x,pose.y)
                    try:
                        path = self.grid.plan(xy,target_xy)
                        if self.axis_routes:
                            path, splits = self.grid.split_axis_segments(xy, path)
                            self.axis_route_splits += splits
                    except RuntimeError as exc:
                        if not self.dynamic_map:
                            raise
                        # A moving fixture can temporarily consume the start
                        # clearance or corridor. Hold and rescan within the
                        # existing bounded navigation budget, never erase it.
                        path=[]
                        self.wait_steps+=1
                        if self.wait_steps==1 or self.wait_steps%25==0:
                            print(f'SENSOR_WAIT obstacle_blocked {exc}',flush=True)
                        continue
                    self.plans += 1
                    self.target_revision += 1
                    self.detours += int(len(path)>1)
                    event=dict(start=xy,goal=target_xy,path=list(path),scan=len(self.scans),
                               invalidations=self.path_invalidations,cleared_cells=self.cleared_cells)
                    if self.peer_positions is not None:
                        event['coordination_keepouts'] = list(self.grid.coordination_keepouts)
                    self.events.append(event)
                    print(f"SENSOR_PLAN scans={len(self.scans)} hits={len(returns)} waypoints={path}",flush=True)
            if yield_command is not None:
                command=bridge.env.unwrapped.command_manager.get_term('base_velocity').command
                command[0,:]=command.new_tensor(yield_command)
                self.range_yield_steps+=1
                bridge._policy_step()
                continue
            if cruise is not None:
                command = bridge.env.unwrapped.command_manager.get_term('base_velocity').command
                for axis, value in enumerate(cruise.command(pose)):
                    command[0, axis] = value
                bridge._policy_step()
                after = bridge.pose()
                self.travel += math.dist(xy, (after.x, after.y))
                if cruise.phase == 'complete':
                    print(f'SENSOR_CRUISE_COMPLETE end={(after.x,after.y)} yaw={after.yaw:.3f}',flush=True)
                    cruise = None
                    path = []  # replan in the original docking orientation
                    cruise_cooldown = self._ticks(100)
                    progress_origin = None
                    burst_steps = lateral_burst_steps = 0
                continue
            while len(path)>1 and math.dist(xy,path[0]) < .20 and self.grid.clear_line(xy,path[1]):
                path.pop(0)
                self.target_revision += 1
            if not path:
                bridge._zero_command()
                bridge._policy_step()
                continue
            cruise_cooldown = max(0, cruise_cooldown-1)
            if self.travel_heading and not cruise_cooldown and step % self._ticks(5) == 0:
                from .travel_heading import ForwardCruise, select_cruise_endpoint
                end = select_cruise_endpoint(pose, path[0], latest_returns, self.grid)
                if end is not None:
                    cruise = ForwardCruise(pose, end, goal.yaw)
                    cruise.check_scan(pose, latest_returns, self.grid)
                    self.cruise_events.append(cruise.event)
                    print(f'SENSOR_CRUISE_START start={xy} end={end} heading={cruise.heading:.3f}',flush=True)
                    bridge._zero_command()
                    bridge._policy_step()
                    continue
            bridge._set_velocity_command(Pose2D(*path[0],goal.yaw))
            # Keep the proven baseline command. A bounded stronger command is
            # used only after measured stagnation, rather than increasing every
            # docking approach (which can leave a tilted trunk on arrival).
            command = bridge.env.unwrapped.command_manager.get_term('base_velocity').command
            forward_limit=.5
            lateral_limit=.35
            command[0,0] = max(-forward_limit,min(forward_limit,float(command[0,0])))
            command[0,1] = max(-lateral_limit,min(lateral_limit,float(command[0,1])))
            body_x_error = math.cos(pose.yaw)*(path[0][0]-pose.x)+math.sin(pose.yaw)*(path[0][1]-pose.y)
            body_y_error = -math.sin(pose.yaw)*(path[0][0]-pose.x)+math.cos(pose.yaw)*(path[0][1]-pose.y)
            forward_primary=not self.vector_tracking or abs(body_x_error)>=abs(body_y_error)
            if self.docking_axis is not None:
                requested=(not transit and math.hypot(body_x_error,body_y_error)<.12
                    and bool(self.docking_axis_scope()))
                forward_primary=self.docking_axis.update(body_x_error,body_y_error,tolerance,
                    self.control_period_s,requested=requested,revision=self.target_revision)
                self.docking_axis_telemetry=self.docking_axis.telemetry
            if progress_origin is None or math.dist(xy,progress_origin)>.04:
                progress_origin=xy
                progress_steps=0
            progress_steps+=1
            if progress_steps>=self._ticks(100) and (
                    self.stalled_forward_command and abs(body_x_error)>.04
                    or self.stalled_lateral_command and abs(body_y_error)>.04):
                burst_steps=self._ticks(40) if self.stalled_forward_command and abs(body_x_error)>.04 else 0
                lateral_burst_steps=self._ticks(40) if self.stalled_lateral_command and abs(body_y_error)>.04 else 0
                progress_steps=0
                self.motion_stall_bursts+=1
                self.lateral_stall_bursts+=int(lateral_burst_steps>0)
                print(f'SENSOR_MOTION_STALL burst={self.motion_stall_bursts} pose={xy}',flush=True)
            minimum=self.stalled_forward_command if burst_steps else self.minimum_forward_command
            if body_x_error>0 and self.minimum_positive_forward_command is not None:
                minimum=max(minimum,self.minimum_positive_forward_command)
            if forward_primary and abs(body_x_error)>min(.04,tolerance*.5) and abs(float(command[0,0]))<minimum:
                command[0,0] = math.copysign(minimum,body_x_error)
            # A few centimetres of cross-axis drift must not inject a full
            # 0.30 m/s reversal into a long lateral traverse. Precise lateral
            # docking still retains its existing floor on the dominant axis.
            if (self.vector_tracking and not forward_primary
                    and math.hypot(body_x_error,body_y_error)<max(.12,2*bridge.config.position_tolerance)
                    and abs(body_y_error)>min(.04,tolerance*.5) and abs(float(command[0,1]))<.22):
                command[0,1]=math.copysign(.22,body_y_error)
            burst_steps=max(0,burst_steps-1)
            if lateral_burst_steps and (not self.vector_tracking or not forward_primary) and abs(body_y_error)>min(.04,tolerance*.5):
                command[0,1] = math.copysign(max(abs(float(command[0,1])),self.stalled_lateral_command),body_y_error)
            lateral_burst_steps=max(0,lateral_burst_steps-1)
            bridge._policy_step()
            after = bridge.pose()
            self.travel += math.dist(xy,(after.x,after.y))
            if step % self._ticks(200) == 0:
                print(f"SENSOR_PROGRESS step={step} pose={xy} goal={target_xy}",flush=True)
        bridge._zero_command()
        self.docking_axis_telemetry=None
        return False

    def finish(self):
        result = dict(sensor='idealized_physx_two_layer_360_range_scan', localization='simulator_ground_truth',
                    dynamic_range_yield=dict(enabled=self.range_yield is not None,
                        active_control_steps=self.range_yield_steps,source='measured range history and robot pose only',
                        enter_distance_m=.85,emergency_distance_m=.62,release_distance_m=.98,
                        speed_mps=.30,swept_lookahead_s=.35),
                    docking_axis_hysteresis=dict(enabled=self.docking_axis is not None,
                        scope='empty original-Z1 receiving, non-transit approach waypoints within .12m',
                        minimum_axis_time_s=.20,switch_dominance_ratio=1.25,
                        active_control_frames=self.docking_axis.active_frames if self.docking_axis else 0,
                        primary_axis_switches=self.docking_axis.switches if self.docking_axis else 0),
                    control_period_s=self.control_period_s,scan_period_s=self._ticks(5)*self.control_period_s,
                    vector_tracking=self.vector_tracking,
                    recovery_budget_s=self._ticks(150)*self.control_period_s,
                    recovery_confirmation_s=self._ticks(10)*self.control_period_s,
                    loaded_recovery_trigger_height_m=self.loaded_recovery_trigger_height,
                      coordination_keepouts=dict(enabled=self.peer_positions is not None,
                          source='centralized peer localization (simulator ground truth)' if self.peer_positions is not None else None,
                          radius_m=1.50,refreshes=self.coordination_refreshes,
                          current_peers=len(self.grid.coordination_keepouts),max_peers=self.coordination_max_peers,
                          blocked_checks=self.grid.coordination_blocked_checks,
                          planning_interventions=self.grid.coordination_plan_interventions),
                      static_map_only=not self.dynamic_map, scans=len(self.scans), plans=self.plans, detour_plans=self.detours,
                      path_invalidations=self.path_invalidations,cleared_cells=self.cleared_cells,
                      blocked_wait_scans=self.wait_steps,
                      stability_pauses=self.stability_pauses,
                      recovery_safety_stops=self.recovery_safety_stops,
                      recovery_sensor_failures=self.recovery_sensor_failures,
                      recovery_payload_suppressed=self.recovery_payload_suppressed,
                      motion_stall_bursts=self.motion_stall_bursts,
                      lateral_stall_bursts=self.lateral_stall_bursts,
                      axis_routes=self.axis_routes,axis_route_splits=self.axis_route_splits,
                      travel_heading=self.travel_heading,
                      forward_cruises=len(self.cruise_events),
                      completed_forward_cruises=sum(e['completed'] for e in self.cruise_events),
                      stalled_lateral_command_mps=self.stalled_lateral_command,
                      minimum_positive_forward_command_mps=self.minimum_positive_forward_command,
                      planning_stops=self.stops, measured_travel_m=self.travel,
                      occupied_returns=len(self.grid.observed))
        measurement_model = getattr(self.scanner, 'measurement_model', None)
        if getattr(self.scanner, 'pose_observation', None) is not None:
            result['localization'] = 'synthetic_perturbed_simulator_pose'
            result['range_projection'] = 'physical range and relative direction projected with cached estimated pose'
        if measurement_model is not None:
            result['sensor'] = 'perturbed_physx_two_layer_360_range_scan'
            result['range_measurement'] = measurement_model.metadata()
            result['range_measurement']['minimum_valid_fraction_per_layer'] = .8
        if self.output_dir:
            (self.output_dir/'navigation.json').write_text(json.dumps(dict(summary=result,events=self.events,
                recovery_events=self.recovery_events,cruise_events=self.cruise_events,scans=self.scans)),encoding='utf-8')
        return result


def configure_navigation_obstacles(cfg):
    """Scene authoring only. This pose is never passed to SensorNavigator."""
    from isaaclab.assets import RigidObjectCfg
    from isaaclab_physx.sensors import ContactSensorCfg
    import isaaclab.sim as sim_utils
    cfg.scene.aisle_blocker = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/AisleBlocker',
        spawn=sim_utils.CuboidCfg(size=(.60,.80,.75),collision_props=sim_utils.CollisionPropertiesCfg(),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=20.),activate_contact_sensors=True,
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.95,.55,.08))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.,.65,.375)))
    cfg.scene.blocker_contact = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/AisleBlocker',update_period=0.,history_length=4)
