"""Materialize a recorded source profile and run its original PowerShell recipe.

--dry-run executes the actual wrapper chain up to the simulator boundary,
captures arguments/environment and exits deliberately before launching Isaac.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def quote_ps(value):return "'"+str(value).replace("'","''")+"'"


def directory_link(source,destination,powershell):
    source=source.resolve();destination=destination.absolute()
    if not source.is_dir():raise FileNotFoundError('Prepare asset directory first: '+str(source))
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    if os.name=='nt':
        statement='New-Item -ItemType Junction -Path '+quote_ps(destination)+' -Target '+quote_ps(source)+' | Out-Null'
        subprocess.run([powershell,'-NoProfile','-NonInteractive','-Command',statement],check=True)
    else:destination.symlink_to(source,target_is_directory=True)


def host_adapter(source):
    """Replace historical machine paths, without touching controller code."""
    start=source.index("$env:CONDA_PREFIX =")
    end=source.index('$fleetResultRequired =',start)
    setup="""$projectRoot = Split-Path -Parent $PSScriptRoot
$isaacLabRoot = $env:GO2_ISAACLAB_ROOT
$isaacSimRoot = $env:GO2_ISAACSIM_ROOT
if ($isaacSimRoot) {
    $env:EXP_PATH = Join-Path $isaacSimRoot 'apps'
    $env:CARB_APP_PATH = Join-Path $isaacSimRoot 'kit'
    $env:ISAAC_PATH = $isaacSimRoot
}

"""
    source=source[:start]+setup+source[end:]
    source=source.replace("Push-Location 'E:\\Robot_Project'",'Push-Location $projectRoot')
    old="& 'C:\\IsaacLab\\isaaclab.bat' -p 'E:\\Robot_Project\\go2_logistics\\isaac_policy_runner.py' @runnerArgs"
    if old not in source:raise RuntimeError('Historical host launcher differs; inspect before adapting')
    new="""if ($env:GO2_RELEASE_LAUNCH_PROBE -eq '1') {
        $captured = @{}
        Get-ChildItem Env: | Where-Object { $_.Name -like 'GO2_*' } | ForEach-Object { $captured[$_.Name] = $_.Value }
        @{arguments=$runnerArgs;environment=$captured;simulationStarted=$false;probeOnly=$true} |
            ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $env:GO2_RELEASE_PROBE_PATH -Encoding UTF8
        exit 42
    }
    & (Join-Path $isaacLabRoot 'isaaclab.bat') -p (Join-Path $PSScriptRoot 'isaac_policy_runner.py') @runnerArgs"""
    return source.replace(old,new)


def materialize(demo,destination):
    if destination.exists():raise FileExistsError('Use a new output directory: '+str(destination))
    package=destination/'go2_logistics'
    shutil.copytree(ROOT/'go2_logistics',package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    overlay=ROOT/'repro'/demo/'overrides'
    if overlay.exists():
        for source in overlay.iterdir():shutil.copy2(source,package/source.name)
    expected=read(ROOT/'repro'/demo/'source_manifest.json')
    for name,digest in expected.items():
        if sha(package/name)!=digest:raise RuntimeError('Frozen source mismatch '+name)
    host=package/'run_go2_mission.ps1'
    original=sha(host);host.write_text(host_adapter(host.read_text(encoding='utf-8-sig')),encoding='utf-8')
    receipt=dict(demo=demo,original_frozen_sources_verified=len(expected),
        untouched_python_sources_verified=sum(n.endswith('.py') for n in expected),
        host_adapter=dict(file=host.name,original_sha256=original,adapted_sha256=sha(host),
            scope='Machine paths and a dry-run exit only; physics/controller Python stays byte-for-byte frozen.'))
    (destination/'source_materialization.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return package


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('demo',choices=['dual_delivery','far_shelf','dynamic_navigation'])
    p.add_argument('--isaaclab',type=Path,default=Path(os.environ.get('ISAACLAB_PATH','C:/IsaacLab')))
    p.add_argument('--isaacsim',type=Path,default=Path(os.environ.get('ISAACSIM_PATH','C:/isaacsim')))
    p.add_argument('--powershell',default=shutil.which('pwsh'))
    p.add_argument('--headless',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not a.powershell:raise SystemExit('PowerShell7 is required; install it or pass --powershell PATH')
    output=(a.output or ROOT/'outputs'/(a.demo+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    package=materialize(a.demo,output/'workspace')
    recipe=read(ROOT/'demos'/a.demo/'recipe.json')
    env={k:v for k,v in os.environ.items() if not k.startswith('GO2_')}
    env.update(GO2_ISAACLAB_ROOT=str(a.isaaclab.resolve()),GO2_ISAACSIM_ROOT=str(a.isaacsim.resolve()),
        PYTHONIOENCODING='utf-8',PYTHONPATH=os.pathsep.join([str(package.parent),
            str(a.isaaclab/'source/isaaclab'),str(a.isaaclab/'source/isaaclab_tasks'),str(a.isaacsim/'site')]))
    if a.dry_run:
        env.update(GO2_RELEASE_LAUNCH_PROBE='1',GO2_RELEASE_PROBE_PATH=str(output/'launch_probe.json'))
    else:
        if os.name!='nt':raise SystemExit('The recorded launcher was validated on Windows; Linux is not validated.')
        if not (a.isaaclab/'isaaclab.bat').is_file():raise FileNotFoundError('Isaac Lab launcher missing')
        for suffix in ['assets','artifacts/go2_asset','artifacts/real_arm_z1','artifacts/isaac_assets']:
            directory_link(ROOT/suffix,package.parent/suffix,a.powershell)
    args=[a.powershell,'-NoProfile','-NonInteractive','-File',str(package/recipe['entrypoint'])]
    values=dict(recipe['parameters'])
    values['Checkpoint']=str(ROOT/recipe['checkpoint'])
    if recipe.get('return_checkpoint'):values['ReturnCheckpoint']=str(ROOT/recipe['return_checkpoint'])
    values['RunDirectory']=str(output/'recording')
    if a.headless or a.dry_run:values['Headless']=True
    for key,value in values.items():
        if isinstance(value,bool):
            if value:args.append('-'+key)
        else:args.extend(['-'+key,str(value)])
    result=subprocess.run(args,env=env,cwd=package.parent)
    if a.dry_run:
        if result.returncode not in (0,42) or not (output/'launch_probe.json').is_file():
            raise SystemExit('Launch probe did not reach the guarded simulator boundary')
        probe=read(output/'launch_probe.json')
        if probe['simulationStarted']:raise RuntimeError('A dry run must never start the simulator')
        print(json.dumps(dict(probe=str(output/'launch_probe.json'),source=read(output/'workspace/source_materialization.json'),
            simulator_started=False),indent=2))
    else:raise SystemExit(result.returncode)


if __name__=='__main__':main()
