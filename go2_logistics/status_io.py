"""A temporarily locked shared UI status must not terminate physics evidence."""
import json
from pathlib import Path
from .run_isolation import publish_json_pointer


def publish_fleet_status(run_dir,status,latest_path=Path('artifacts/fleet/latest_status.json')):
    payload=json.dumps(status,indent=2)
    # This run's own status and full control/mission evidence remain primary.
    (Path(run_dir)/'status.json').write_text(payload,encoding='utf-8')
    return publish_json_pointer(latest_path, status)
