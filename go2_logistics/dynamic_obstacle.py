"""Scripted moving fixture for repeatable navigation evaluation.

Only the obstacle's kinematic body is driven. Its pose/trajectory is never
passed to the scanner/planner. This is not a learned forklift policy.
"""
import json
import math


def fleet_obstacle_position(time_s):
    """A cart crosses the two front aisles, outside the shelf collision boxes.

    This function belongs only to the scenario driver, never navigation.
    Maximum translation is 1.4 mm per original 5 ms physics step.
    """
    if not math.isfinite(time_s) or time_s<0:
        raise ValueError('Obstacle clock must be finite and nonnegative')
    distance=4.6
    travel=(max(0.,time_s-1.5)*.28)%(2.*distance)
    y=-2.3+(travel if travel<=distance else 2.*distance-travel)
    return (-1.55,y,.375)


class FleetCrossingObstacle:
    """Drive only the fixture; observe real rays and contact after physics.

    No pose, schedule or object identity is passed into the planner. Truth
    positions below are isolated evaluation records, not navigation inputs.
    """
    def __init__(self,env,output_dir,names,sample_stride):
        from pathlib import Path
        self.env=env.unwrapped
        self.body=self.env.scene['aisle_blocker']
        self.sensor=self.env.scene.sensors['blocker_contact']
        self.output=Path(output_dir)
        self.names=tuple(names)
        self.sample_stride=sample_stride
        self.elapsed=0.
        self.max_contact_n=0.
        self.max_displacement_m=0.
        self.seen_scans={name:0 for name in names}
        self.last_scans={name:-1 for name in names}
        self.samples=0
        self.stream=(self.output/'dynamic_obstacle_trace.jsonl').open('w',encoding='utf-8')

    def before_physics(self,dt):
        import torch
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('Moving fixture requires a finite positive physics period')
        self.pending_elapsed=self.elapsed+dt
        position=fleet_obstacle_position(self.pending_elapsed)
        pose=torch.tensor([[*position,0.,0.,0.,1.]],device=self.env.device)
        self.body.write_root_pose_to_sim(pose)

    def after_physics(self,robots,navigators,phases,commands,tick):
        import torch
        self.elapsed=getattr(self,'pending_elapsed',self.elapsed)
        def tensor(x):return getattr(x,'torch',x)
        position=tensor(self.body.data.root_pos_w)[0].cpu().tolist()
        contact=float(torch.linalg.vector_norm(tensor(self.sensor.data.net_forces_w),dim=-1).max())
        if not math.isfinite(contact):raise RuntimeError('Moving fixture contact measurement is nonfinite')
        self.max_contact_n=max(self.max_contact_n,contact)
        self.max_displacement_m=max(self.max_displacement_m,math.dist(position,fleet_obstacle_position(0.)))
        observations={}
        for name in self.names:
            scanner=navigators[name].scanner
            hits=sum(count for path,count in getattr(scanner,'last_hit_paths',{}).items() if '/AisleBlocker' in path)
            if scanner.scan_number!=self.last_scans[name]:
                self.seen_scans[name]+=int(hits>0)
                self.last_scans[name]=scanner.scan_number
            root=tensor(robots[name].data.root_pos_w)[0].cpu().tolist()
            # Distance from base center to the actual .60 x .80 m fixture.
            # This is not a full robot-body clearance or a planner input.
            gap=[max(0.,abs(root[i]-position[i])-(.3,.4)[i]) for i in (0,1)]
            observations[name]=dict(root_position_w=root,phase=phases[name],
                command=commands[name][0].cpu().tolist(),scan=scanner.scan_number,
                nearest_fixture_ray_hits=hits,base_center_to_fixture_xy_m=math.hypot(*gap))
        if tick%self.sample_stride==0 or contact>1.:
            self.stream.write(json.dumps(dict(tick=tick,sim_time_s=self.elapsed,
                fixture_position_w=position,fixture_contact_n=contact,robots=observations))+'\n')
            self.samples+=1
            if self.samples%25==0:self.stream.flush()
        if contact>1.:
            raise RuntimeError(f'Moving aisle fixture contact exceeded 1 N: {contact:.6f} N')

    def close(self):
        import hashlib
        self.stream.close()
        report=dict(enabled=True,scenario='Scripted kinematic cart crossing both front aisles',
            navigation_inputs='Existing PhysX ray ranges only; fixture pose/clock/identity are evaluation-only',
            not_slam=True,not_learned_obstacle_policy=True,
            speed_mps=.28,size_m=[.60,.80,.75],elapsed_s=self.elapsed,
            maximum_fixture_contact_n=self.max_contact_n,contact_limit_n=1.,
            maximum_displacement_m=self.max_displacement_m,observed_scans=self.seen_scans,
            trace_samples=self.samples,trace_sha256=hashlib.sha256((self.output/'dynamic_obstacle_trace.jsonl').read_bytes()).hexdigest(),
            exposure_verified=self.max_displacement_m>.5 and all(n>0 for n in self.seen_scans.values()),
            scope='Dynamic navigation exposure only; full success also requires original robot, route and home guards')
        (self.output/'dynamic_obstacle.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report


class CrossingObstacle:
    def __init__(self,env):
        self.env=env.unwrapped
        self.body=self.env.scene['aisle_blocker']
        self.elapsed=None
        self.events=[]
        self.step_count=0
        self.position=[-1.,-1.35,.375]

    def step(self,arm,output_dir):
        import torch
        self.step_count+=1
        root=getattr(arm.robot.data.root_pos_w,'torch',arm.robot.data.root_pos_w)[0]
        if self.elapsed is None:
            if not (arm.held and arm.state=='navigate_dropoff' and float(root[0])>2 and float(root[1])<1.):
                return
            self.elapsed=0.
            self.events.append(dict(event='obstacle_entering_aisle',step=self.step_count))
            print('DYNAMIC_OBSTACLE entering_aisle',flush=True)
        self.elapsed+=self.env.step_dt
        # 0.20 m/s entering, hold, then withdraw. Continuous position target.
        t=self.elapsed
        y=-1.35+.20*min(t,10.) if t<22 else .65-.20*min(t-22,10.)
        self.position=[-1.,y,.375]
        pose=torch.tensor([[-1.,y,.375,0.,0.,0.,1.]],device=self.env.device)
        self.body.write_root_pose_to_sim(pose)
        if t>=32 and len(self.events)==1:
            self.events.append(dict(event='obstacle_withdrawn',step=self.step_count))
            print('DYNAMIC_OBSTACLE withdrawn',flush=True)
        if self.step_count%25==0:
            (output_dir/'dynamic_obstacle.json').write_text(json.dumps(dict(events=self.events,
                elapsed_s=t,position=self.position,source='scripted_kinematic_fixture'),indent=2),encoding='utf-8')
