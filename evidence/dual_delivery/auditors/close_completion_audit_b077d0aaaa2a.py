"""Replay each warehouse parcel's opt-in early closing independently."""
import json
from pathlib import Path

from .z1_ground_pickup_audit import audit_close_completion


def replay(rows,configs):
    enabled={name for name,cfg in configs.items() if cfg.get('close_on_stable_contact') is True}
    groups={};issues=[]
    for row in rows:
        for name in enabled:
            robot=row['robots'][name];arm=robot.get('original_z1') or {}
            code=arm.get('active_payload_code')
            if not code:continue
            key=(name,code)
            if arm.get('state')=='closing_fingers':
                group=groups.setdefault(key,dict(rows=[],completion=None))
                group['rows'].append(dict(tick=row['tick'],robots=dict(dog1=dict(state='closing_fingers',
                    jaw_normal_n=arm['jaw_normal_n'],jaw_total_n=arm['jaw_total_n']))))
            if key in groups and arm.get('close_completion') is not None:
                groups[key]['completion']=arm['close_completion']
    orders=[]
    for (name,code),group in groups.items():
        report=audit_close_completion(group['rows'],dict(config=configs[name],close_completion=group['completion']))
        orders.append(dict(robot=name,code=code,**report))
    if enabled and not orders:issues.append('No completed closing evidence for enabled experiment')
    return dict(enabled=bool(enabled),passed=not issues and all(r['passed'] for r in orders),
        issues=issues,orders=orders,scope='Per-parcel measured closing timing; no navigation or delivery acceptance')


def audit_warehouse_close(folder,result):
    configs=result.get('manipulation_config') or {}
    if not any(c.get('close_on_stable_contact') is True for c in configs.values()):
        return dict(enabled=False,passed=True)
    with (Path(folder)/'control_trace.jsonl').open(encoding='utf-8') as stream:
        return replay((json.loads(line) for line in stream),configs)
