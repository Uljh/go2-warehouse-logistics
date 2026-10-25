"""Visual two-layer receiving gate independent of the physics backend."""
import math


class ReceivingReadiness:
    """Upright single-slot pickup requires a stopped belt and repeated RGB-D stability."""
    def __init__(self, bin_xy, required_samples=3):
        self.bin_xy=bin_xy
        self.required_samples=required_samples
        self.previous=None
        self.count=0

    def observe(self, detection, *, conveyor_running):
        center=(detection or {}).get('center_world')
        code=(detection or {}).get('parcel_code')
        valid=(not conveyor_running and center is not None and len(center)==3
               and all(math.isfinite(v) for v in center) and bool(code)
               and math.dist(center[:2],self.bin_xy)<.05 and .035<center[2]<.12)
        if not valid:
            self.previous=None
            self.count=0
            return False
        if self.previous and self.previous[0]==code and math.dist(self.previous[1],center)<.002:
            self.count+=1
        else:
            self.count=1
        self.previous=(code,tuple(center))
        return self.count>=self.required_samples


class StackReadiness:
    def __init__(self, bin_xy, required_samples=3):
        self.bin_xy=bin_xy
        self.required_samples=required_samples
        self.previous=None
        self.count=0

    def observe(self, detection):
        geometry=(detection or {}).get('geometry') or {}
        center=geometry.get('center_world')
        code=(detection or {}).get('parcel_code')
        valid=(center is not None and len(center)==3 and all(math.isfinite(v) for v in center)
               and bool(code) and abs(center[0]-self.bin_xy[0])<.025
               and abs(center[1]-self.bin_xy[1])<.025
               and .13<center[2]<.15 and .075<geometry.get('height_m',0)<.085)
        if not valid:
            self.previous=None
            self.count=0
            return False
        if self.previous and self.previous[0]==code and math.dist(self.previous[1],center)<.002:
            self.count+=1
        else:
            self.count=1
        self.previous=(code,tuple(center))
        return self.count>=self.required_samples
