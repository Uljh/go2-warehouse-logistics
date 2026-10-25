param(
    [int]$NumEnvs = 256,
    [int]$Iterations = 150,
    [double]$EvalSeconds = 40,
    [switch]$EvalOnly,
    [ValidateSet('panda_surrogate','z1')][string]$ArmModel = 'panda_surrogate',
    [string]$Checkpoint = 'E:/Robot_Project/artifacts/go2_official_checkpoint.pt',
    [string]$PoseSource = 'E:/Robot_Project/artifacts/fleet/run_20261019_171441_849946/fleet_result.json',
    [string]$Z1PoseTrace = '',
    [string[]]$CommandTrace = @()
)
$ErrorActionPreference = 'Stop'
$env:CONDA_PREFIX = 'C:/Users/admin/.conda/envs/env_isaaclab'
$env:EXP_PATH = 'C:/isaacsim/apps'
$env:CARB_APP_PATH = 'C:/isaacsim/kit'
$env:ISAAC_PATH = 'C:/isaacsim'
$env:PYTHONPATH = 'E:/Robot_Project;C:/isaacsim/site;C:/IsaacLab/source/isaaclab;C:/IsaacLab/source/isaaclab_tasks;'
$env:PYTHONUNBUFFERED = '1'
$trainingArguments = @('-m','go2_logistics.train_mounted_gait','--num-envs',"$NumEnvs",'--iterations',"$Iterations",'--eval-seconds',"$EvalSeconds",'--checkpoint',$Checkpoint,'--pose-source',$PoseSource,'--viz','none','--device','cuda:0')
$trainingArguments += @('--arm-model',$ArmModel)
if ($Z1PoseTrace) { $trainingArguments += @('--z1-pose-trace',$Z1PoseTrace) }
foreach ($tracePath in $CommandTrace) { $trainingArguments += @('--command-trace', $tracePath) }
if ($EvalOnly) { $trainingArguments += '--eval-only' }
# Windows PowerShell treats native stderr warnings as errors when redirected.
# Python's exit code is authoritative; simulator warnings must not kill it.
$ErrorActionPreference = 'Continue'
& C:/Users/admin/.conda/envs/env_isaaclab/python.exe @trainingArguments
exit $LASTEXITCODE
