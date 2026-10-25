"""Record checkpoint identity without inferring training from its filename."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OFFICIAL_SHA256 = '32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0'


def checkpoint_provenance(checkpoint):
    path = Path(checkpoint).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(path=str(path), sha256=digest, kind='external_checkpoint',
                  training_manifest_verified=False)
    if digest == OFFICIAL_SHA256:
        result['kind'] = 'nvidia_official_pretrained'
        return result
    manifest_path = path.parent/'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('actual_training_completed') and manifest.get('trained_checkpoint_sha256') == digest:
            result.update(kind='project_mounted_gait_finetuned', training_manifest_verified=True,
                training_manifest=str(manifest_path),
                source_checkpoint_sha256=manifest.get('source_checkpoint_sha256'),
                training_scope=manifest.get('training_scope'),
                environment_transitions=manifest.get('environment_transitions'),
                evaluation_survived=manifest.get('evaluation_survived'))
    return result
