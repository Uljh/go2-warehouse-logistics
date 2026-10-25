"""Per-experiment writable paths and best-effort discovery pointers.

Consumers must validate the result inside their own run directory. Shared
``latest`` files exist only for the UI and may point at any concurrent run.
"""
from pathlib import Path
from datetime import datetime
import json
import tempfile
import uuid


def create_run_directory(root, requested=None, *, prefix='run'):
    """Reserve a fresh directory; reject reuse so old evidence cannot leak in."""
    path = (Path(requested) if requested is not None else
            Path(root) / f'{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}')
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def publish_json_pointer(path, value):
    """Atomically publish a UI pointer using a writer-private temporary file.

    A locked UI pointer is nonfatal. Errors on the authoritative run files
    must still propagate at their call site.
    """
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                prefix=path.name + '.', suffix='.tmp', dir=path.parent,
                delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
        temporary.replace(path)
        return True
    except PermissionError:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError:
                pass
