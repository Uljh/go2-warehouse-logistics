"""Range-only active yielding when a moving surface approaches the robot.

No fixture pose, trajectory or identity enters this controller. This is a
bounded reactive layer, not SLAM, object tracking or a learned navigation
policy. Normal sensed-map planning resumes after measured clearance returns.
"""
import math


class RangeYield:
    def __init__(self):
        self.previous=None
        self.active=False
        self.command=None
        self.telemetry=None

    def update(self,pose,returns,dt,*,goal=None,bounds=None):
        if not math.isfinite(dt) or dt<=0:raise ValueError('Range yield requires a positive scan period')
        points=[(x,y) for x,y,layer,_ in returns if layer in (0,1) and math.isfinite(x) and math.isfinite(y)]
        if not points:
            self.previous=None;self.active=False;self.command=None
            self.telemetry=dict(active=False,reason='no_observed_surface')
            return None
        xy=(pose.x,pose.y)
        point=min(points,key=lambda p:math.dist(xy,p))
        distance=math.dist(xy,point)
        n=((point[0]-pose.x)/max(distance,1.e-9),(point[1]-pose.y)/max(distance,1.e-9))
        closing=0.;associated=False
        if self.previous is not None:
            old_xy,old_point,old_distance=self.previous
            associated=math.dist(point,old_point)<.25
            if associated:
                own_approach=sum((xy[i]-old_xy[i])*n[i] for i in (0,1))/dt
                closing=max(0.,min(.6,(old_distance-distance)/dt-own_approach))
        self.previous=(xy,point,distance)
        # Leave approach and velocity-tracking delay before the body margin
        # is consumed. At <.95m yield even if association is ambiguous. This does not
        # suppress any existing contact, posture, traffic or recovery guard.
        requested=(distance<1.25 and closing>.025) or distance<.95
        self.active=bool(requested or self.active and distance<1.40)
        self.command=None
        selected=None
        if self.active:
            direction=math.atan2(-n[1],-n[0])
            candidates=[]
            offsets=(0.,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3)
            if goal is not None:offsets+= (math.pi/2,-math.pi/2)
            for offset in offsets:
                velocity=(.30*math.cos(direction+offset),.30*math.sin(direction+offset))
                end=(xy[0]+velocity[0]*.35,xy[1]+velocity[1]*.35)
                # Stay in the known planning domain, leaving room to brake.
                # Bounds contain no obstacle pose or motion information.
                if bounds is not None and not (bounds[0]+.20<=end[0]<=bounds[1]-.20
                        and bounds[2]+.20<=end[1]<=bounds[3]-.20):
                    continue
                if not self._swept_clear(xy,end,points):continue
                # Predict only the associated approaching surface, using its
                # range-derived radial speed. Reject a side-step that consumes
                # the clearance within one second. Other returns stay in the
                # swept-space check; no fixture identity/trajectory is used.
                relative=(-closing*n[0]-velocity[0],-closing*n[1]-velocity[1])
                r=(point[0]-xy[0],point[1]-xy[1])
                speed2=sum(v*v for v in relative)
                closest_t=max(0.,min(1.,-sum(r[i]*relative[i] for i in (0,1))/max(speed2,1.e-12)))
                predicted=math.hypot(*(r[i]+closest_t*relative[i] for i in (0,1)))
                if predicted<min(.80,distance)-1.e-9:continue
                radial=-sum(velocity[i]*n[i] for i in (0,1))
                progress=0. if goal is None else (math.dist(xy,goal)-math.dist(end,goal))/.35
                # Prefer increasing clearance. Goal progress only breaks near
                # ties: the first goal-dominant physical trial cut the corner
                # of the moving box and made real contact. The boundary filter
                # still forces a safe side-step before leaving the map.
                score=radial+.10*progress if goal is not None else -abs(offset)
                candidates.append((score,velocity,predicted))
            if candidates:
                _,velocity,predicted=max(candidates,key=lambda item:item[0])
                c,s=math.cos(pose.yaw),math.sin(pose.yaw)
                self.command=(c*velocity[0]+s*velocity[1],-s*velocity[0]+c*velocity[1],0.)
                selected=dict(world_velocity=velocity,predicted_nearest_distance_m=predicted,
                    goal=goal,bounds=bounds)
            if self.command is None:self.command=(0.,0.,0.)
        self.telemetry=dict(active=self.active,nearest_distance_m=distance,
            observed_surface=point,estimated_surface_closing_mps=closing,associated=associated,
            command=self.command,selection=selected,
            reason='range_surface_approaching' if requested else 'clearance_hysteresis' if self.active else 'clear')
        return self.command

    @staticmethod
    def _swept_clear(start,end,points):
        vx,vy=end[0]-start[0],end[1]-start[1]
        length2=vx*vx+vy*vy
        for point in points:
            t=max(0.,min(1.,((point[0]-start[0])*vx+(point[1]-start[1])*vy)/max(length2,1.e-12)))
            closest=(start[0]+t*vx,start[1]+t*vy)
            # Already-intruding points may only recede, never get closer.
            required=min(.52,math.dist(start,point))
            if math.dist(closest,point)<required-1.e-9:return False
        return True
