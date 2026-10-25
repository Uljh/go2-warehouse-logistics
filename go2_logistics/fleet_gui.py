"""Keep the fleet mission clock separate from Kit's viewport update loop.

UI callbacks enqueue requests. SimulationContext.play/pause pump Kit themselves,
so applying them inside a Kit callback would recursively enter the renderer.
"""

class FleetGuiControl:
    PLAY_SETTING = '/app/player/playSimulations'

    def __init__(self, sim, app, timeline, *, wait_for_start=False):
        self.sim, self.app, self.timeline = sim, app, timeline
        self.state = 'waiting_for_start' if wait_for_start else 'running'
        self._pending_toggles = 0
        self._pending_camera = None
        self._initialized = False

    def request_toggle(self):
        if self.state != 'finished':
            self._pending_toggles += 1

    def request_camera(self, mode):
        self._pending_camera = mode

    def take_camera_request(self):
        mode, self._pending_camera = self._pending_camera, None
        return mode

    def _without_kit_physics(self, operation):
        previous = self.sim.get_setting(self.PLAY_SETTING)
        self.sim.set_setting(self.PLAY_SETTING, False)
        try:
            operation()
        finally:
            self.sim.set_setting(self.PLAY_SETTING, previous if previous is not None else True)

    def pump_ui(self):
        # Raw app.update() with playSimulations enabled can step PhysX outside
        # env.step(), invalidating the shared-world clock and finished evidence.
        self._without_kit_physics(self.app.update)

    def synchronize(self):
        if self.state != 'finished' and self.timeline.is_stopped():
            raise RuntimeError('Timeline stopped; relaunch to rebuild physics')
        changed = bool(self._pending_toggles)
        while self._pending_toggles:
            self._pending_toggles -= 1
            self.state = 'paused' if self.state == 'running' else 'running'
        if self._initialized and not changed:
            if self.state == 'running' and not self.timeline.is_playing():
                self.state = 'paused'
            elif self.state == 'paused' and self.timeline.is_playing():
                self.state = 'running'
        # The initial waiting gate accepts the panel START button only. Kit
        # startup PLAY events or toolbar Play cannot silently start a mission.
        playing = self.state == 'running'
        if playing:
            if not self.sim.is_playing() or not self.timeline.is_playing():
                self._without_kit_physics(self.sim.play)
        elif self.sim.is_playing() or self.timeline.is_playing():
            self._without_kit_physics(self.sim.pause)
        self._initialized = True
        return playing and self.sim.is_playing() and self.timeline.is_playing()

    def finish(self):
        self.state = 'finished'
        self._pending_toggles = 0
        self.synchronize()
