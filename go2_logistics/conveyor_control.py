"""One-slot receiving backpressure using a simulated photoelectric beam.

The beam reports occupancy only, never a parcel identity or destination.
Surface velocity is switched; no parcel body poses are commanded.
"""
class ReceivingGate:
    def __init__(self,env,pickup):
        from omni.physx import get_physx_scene_query_interface
        import omni.usd
        from pxr import PhysxSchema
        self.env=env.unwrapped
        self.pickup=pickup
        self.query=get_physx_scene_query_interface()
        prim=omni.usd.get_context().get_stage().GetPrimAtPath('/World/envs/env_0/Conveyor')
        self.api=PhysxSchema.PhysxSurfaceVelocityAPI(prim)
        self.reset()

    def reset(self):
        self.occupied_steps=0
        self.running=True
        self.events=[]
        self.steps=0
        self.api.GetSurfaceVelocityEnabledAttr().Set(True)
        self.stream_audit = getattr(self, 'stream_audit', None)
        self.stream_objects = getattr(self, 'stream_objects', {})

    def attach_stream_audit(self, audit, objects, *, tick=0, sample_interval=10):
        """Attach a finite-feed ledger and sample its rigid bodies each frame.

        ``objects`` maps decoded parcel codes to Isaac rigid-object handles.
        The handles are used only for measured position evidence; they never
        supply identity to the dispatcher.  The queue is physically authored on
        the belt before the first step, so attaching it records a release event
        without moving or resetting any body.
        """
        if set(objects) != set(audit.codes):
            raise ValueError('stream audit object map must cover each unique feed code')
        if not isinstance(sample_interval, int) or sample_interval < 1:
            raise ValueError('stream audit sample_interval must be a positive integer')
        self.stream_audit = audit
        self.stream_objects = dict(objects)
        self.stream_sample_interval = sample_interval
        for code in audit.codes:
            audit.mark_on_belt(code, tick=tick)
        return audit

    def _record_stream_positions(self):
        if self.stream_audit is None or self.steps % self.stream_sample_interval:
            return
        for code, obj in self.stream_objects.items():
            try:
                position = obj.data.root_pos_w[0].detach().cpu().tolist()
            except AttributeError:
                position = obj.data.root_pos_w[0].cpu().tolist()
            self.stream_audit.record_position(code, position, tick=self.steps)

    def step(self,arm,output_dir):
        self.steps+=1
        occupied=False
        def report(hit):
            nonlocal occupied
            path=str(hit.get('rigid_body') if isinstance(hit,dict) else hit.rigid_body)
            if '/Parcel' in path:
                occupied=True
            return True
        self.query.raycast_all((self.pickup.x-.03,self.pickup.y-.12,.07),(0.,1.,0.),.24,report)
        self._record_stream_positions()
        self.occupied_steps=self.occupied_steps+1 if occupied else 0
        desired=not occupied if not self.running else self.occupied_steps*self.env.step_dt<.8
        if desired!=self.running:
            import json
            self.running=desired
            self.api.GetSurfaceVelocityEnabledAttr().Set(desired)
            self.events.append(dict(event='belt_resume' if desired else 'belt_pause_bin_occupied',step=self.steps,
                                    source='idealized_photoelectric_ray'))
            (output_dir/'feed_events.json').write_text(json.dumps(self.events,indent=2),encoding='utf-8')
            print(f'RECEIVING_GATE running={desired} step={self.steps}',flush=True)
