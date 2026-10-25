param(
    [int]$MaxControlSteps = 600,
    [string]$RunDirectory = '',
    [int]$Seed = 7,
    [ValidateSet('assisted','contact')][string]$GraspMode = 'assisted',
    [switch]$GroundBin,
    [switch]$Sorting,
    [switch]$Obstacles,
    [switch]$Conveyor,
    [switch]$WarehouseDemo,
    [switch]$DynamicObstacle,
    [switch]$Continuous,
    [switch]$StreamingFeed,
    [ValidateSet(0,2)][int]$Fleet = 0,
    [ValidateSet('panda_surrogate','z1')][string]$ArmModel = 'panda_surrogate',
    [switch]$FleetSmoke,
    [switch]$FleetGuiCheck,
    [ValidateSet('policy','force_balance')][string]$FleetRecoveryController = '',
    [ValidateSet('legacy','settled_slew')][string]$FleetRecoveryHandoff = 'legacy',
    [switch]$FleetLateralStall,
    [switch]$FleetDetailVideo,
    [ValidateRange(0,1000)][int]$FleetEnduranceCycles = 0,
    [int]$FleetTicks = 30000,
    [switch]$IndoorWarehouse,
    [switch]$RecoverDrops,
    [switch]$InjectDrop,
    [switch]$VisualGeometry,
    [switch]$WideGripper,
    [switch]$OcclusionDemo,
    [double]$ParcelRoll = 0,
    [double]$ParcelYaw = 0,
    [string]$BatchCodes = '',
    [string]$ParcelCode = '1-001',
    [string]$Checkpoint = 'E:\Robot_Project\artifacts\go2_official_checkpoint.pt',
    [double]$PickupX = 4.0,
    [double]$PickupY = 1.5,
    [double]$DropoffX = -3.0,
    [double]$DropoffY = 1.5,
    [switch]$Offline,
    [switch]$AllowRemoteAssets,
    [switch]$Visual,
    [switch]$LocomotionOnly,
    [switch]$WaitForStart,
    [string]$WarehouseScene = 'E:\Robot_Project\artifacts\go2_warehouse.usd'
)

$env:CONDA_PREFIX = 'C:\Users\admin\.conda\envs\env_isaaclab'
$env:EXP_PATH = 'C:\isaacsim\apps'
$env:CARB_APP_PATH = 'C:\isaacsim\kit'
$env:ISAAC_PATH = 'C:\isaacsim'
$env:PYTHONPATH = "E:\Robot_Project;C:\isaacsim\site;C:\IsaacLab\source\isaaclab;C:\IsaacLab\source\isaaclab_tasks;"

$fleetResultRequired = -not $Offline -and ($Fleet -or $FleetSmoke -or $FleetGuiCheck -or $FleetEnduranceCycles)
$previousLaunchToken = $env:GO2_MISSION_RUN_TOKEN
$previousRecoveryController = $env:GO2_FLEET_RECOVERY_CONTROLLER
$fleetOptionEnv = @{}
if ($PSBoundParameters.ContainsKey('FleetRecoveryHandoff')) { $fleetOptionEnv['GO2_BALANCE_HANDOFF'] = $FleetRecoveryHandoff }
if ($PSBoundParameters.ContainsKey('FleetLateralStall')) { $fleetOptionEnv['GO2_FLEET_LATERAL_STALL'] = [string][int]$FleetLateralStall.IsPresent }
if ($PSBoundParameters.ContainsKey('FleetDetailVideo')) { $fleetOptionEnv['GO2_FLEET_DETAIL_VIDEO'] = [string][int]$FleetDetailVideo.IsPresent }
$previousFleetOptionEnv = @{}
foreach ($optionName in $fleetOptionEnv.Keys) {
    $previousFleetOptionEnv[$optionName] = [Environment]::GetEnvironmentVariable($optionName, 'Process')
    [Environment]::SetEnvironmentVariable($optionName, $fleetOptionEnv[$optionName], 'Process')
}
if ($FleetRecoveryController) {
    $env:GO2_FLEET_RECOVERY_CONTROLLER = $FleetRecoveryController
}
if ($fleetResultRequired) {
    . (Join-Path $PSScriptRoot 'check_fleet_result.ps1')
    $launchToken = [guid]::NewGuid().ToString('N')
    $env:GO2_MISSION_RUN_TOKEN = $launchToken
    if (-not $RunDirectory) {
        $RunDirectory = Join-Path 'E:\Robot_Project\artifacts\fleet' ('run_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $launchToken)
    }
}
Push-Location 'E:\Robot_Project'
try {
    $runnerArgs = @(
        '--checkpoint', $Checkpoint,
        '--steps', $MaxControlSteps,
        '--seed', $Seed,
        '--grasp-mode', $GraspMode,
        '--arm-model', $ArmModel,
        '--pickup', $PickupX, $PickupY,
        '--dropoff', $DropoffX, $DropoffY
    )
    if ($RunDirectory) { $runnerArgs += @('--run-dir', $RunDirectory) }
    if ($Offline) {
        $runnerArgs += '--offline'
    }
    if ($LocomotionOnly) { $runnerArgs += '--locomotion-only' }
    if ($GroundBin) { $runnerArgs += '--ground-bin' }
    if ($Obstacles) { $runnerArgs += '--obstacles' }
    if ($Conveyor) { $runnerArgs += '--conveyor' }
    if ($WarehouseDemo) { $runnerArgs += '--warehouse-demo' }
    if ($DynamicObstacle) { $runnerArgs += '--dynamic-obstacle' }
    if ($Continuous) { $runnerArgs += '--continuous' }
    if ($StreamingFeed) { $runnerArgs += '--streaming-feed' }
    if ($Fleet) { $runnerArgs += @('--fleet', $Fleet, '--fleet-ticks', $FleetTicks) }
    if ($FleetSmoke) { $runnerArgs += @('--fleet-smoke', '--fleet-ticks', $FleetTicks) }
    if ($FleetGuiCheck) { $runnerArgs += @('--fleet-gui-check', '--fleet-ticks', $FleetTicks) }
    if ($FleetEnduranceCycles) { $runnerArgs += @('--fleet-endurance-cycles', $FleetEnduranceCycles, '--fleet-ticks', $FleetTicks) }
    if ($IndoorWarehouse) { $runnerArgs += '--indoor-warehouse' }
    if ($RecoverDrops) { $runnerArgs += '--recover-drops' }
    if ($InjectDrop) { $runnerArgs += '--inject-drop' }
    if ($VisualGeometry) { $runnerArgs += @('--visual-geometry','--parcel-roll',$ParcelRoll,'--parcel-yaw',$ParcelYaw) }
    if ($OcclusionDemo) { $runnerArgs += '--occlusion-demo' }
    if ($WideGripper) { $runnerArgs += '--wide-gripper' }
    if ($BatchCodes) { $runnerArgs += @('--batch-codes',$BatchCodes) }
    if ($Sorting) { $runnerArgs += @('--sorting', '--parcel-code', $ParcelCode) }
    if ($AllowRemoteAssets) {
        $runnerArgs += '--allow-remote-assets'
    }
    if ($Visual) {
        if ($Fleet -or $FleetSmoke -or $FleetEnduranceCycles) {
            if ($WaitForStart) {
                Write-Host 'Shared-world fleet waits after loading. Use START / RESUME to begin; close the app to exit.'
            } else {
                Write-Host 'Shared-world fleet starts after loading. Use PAUSE / RESUME; close the app to exit. Completed scenes remain visible.'
            }
        } else {
            Write-Host 'Go2 policy will start automatically after loading. Use PAUSE MISSION / RESUME MISSION / RUN AGAIN in the mission panel.'
        }
        $runnerArgs += '--visual'
        $runnerArgs += '--viz'
        $runnerArgs += 'kit'
        $runnerArgs += '--warehouse-scene'
        $runnerArgs += $WarehouseScene
        if ($WaitForStart) {
            $runnerArgs += '--wait-for-start'
        }
    } else {
        $runnerArgs += '--headless'
    }
    & 'C:\IsaacLab\isaaclab.bat' -p 'E:\Robot_Project\go2_logistics\isaac_policy_runner.py' @runnerArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab mission failed with exit code $LASTEXITCODE"
    }
    if ($fleetResultRequired) {
        Assert-FleetMissionResult -ResultPath (Join-Path $RunDirectory 'fleet_result.json') -LaunchToken $launchToken -RequireGuiCheck:$FleetGuiCheck
    }
}
finally {
    Pop-Location
    if ($fleetResultRequired) {
        $env:GO2_MISSION_RUN_TOKEN = $previousLaunchToken
    }
    if ($FleetRecoveryController) {
        $env:GO2_FLEET_RECOVERY_CONTROLLER = $previousRecoveryController
    }
    foreach ($optionName in $previousFleetOptionEnv.Keys) {
        [Environment]::SetEnvironmentVariable($optionName, $previousFleetOptionEnv[$optionName], 'Process')
    }
}
