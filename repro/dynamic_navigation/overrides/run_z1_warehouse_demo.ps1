# Original-mass Z1 candidate. Finite feed; not a claim of onboard SLAM.
param(
    [switch]$Headless,
    [switch]$StreamingFeed,
    [switch]$DynamicObstacle,
    [string]$RunDirectory = '',
    [string]$Checkpoint = 'E:/Robot_Project/artifacts/gait_adaptation/run_20261022_185754/mounted_gait.pt',
    [int]$Seed = 8,
    [string]$BatchCodes = '',
    [ValidateRange(0.44, 0.52)][double]$CarryLift = 0.48,
    [ValidateRange(0.14, 0.16)][double]$JawSampleX = 0.15,
    [ValidateRange(0.0, 0.50)][double]$ReturnSpeedLimit = 0.0,
    [ValidateRange(0.285, 0.305)][double]$LoadedRecoveryTriggerHeight = 0.285,
    [switch]$DepartureClearance,
    [switch]$SettleStow,
    [switch]$StowBrakeSupport,
    [switch]$LoadedBrakeSupport,
    [switch]$ReceivingBrakeSupport,
    [switch]$PlacementWrenchQP,
    [switch]$LoadedBrakeWrenchQP,
    [switch]$TotalMotorWrenchQP,
    [switch]$JointMotorWrenchQP,
    [switch]$LoadedWaitWrenchQP,
    [switch]$ManipulationMotorQP,
    [switch]$LoadedSupportHysteresis,
    [switch]$AirborneTouchdown,
    [switch]$ContactWrench,
    [switch]$PipelineReceiving,
    [switch]$DockingAxisHysteresis,
    [switch]$CompactReceiving,
    [ValidateRange(2.4, 3.1)][double]$ReceivingApproachX = 2.4,
    [switch]$ReceivingApproachStagedOnly,
    [switch]$SettleReceiving,
    [switch]$ShortSpineReservation,
    [switch]$SettleReservation,
    [ValidateRange(0.0, 2.0)][double]$HandoffReleaseSeconds = 0.0,
    [string]$ReturnCheckpoint = '',
    [string]$CarryCheckpoint = '',
    [switch]$EmptyTransitSpecialist,
    [switch]$ResourceWaitSpecialist,
    [switch]$RepeatPickupSpecialist,
    [switch]$SpecialistIdleControl,
    [ValidateRange(4.0, 8.0)][double]$ApproachSeconds = 8.0,
    [ValidateRange(4.0, 8.0)][double]$PregraspSeconds = 8.0,
    [ValidateRange(8.0, 12.0)][double]$LiftSeconds = 12.0,
    [ValidateRange(8.0, 24.0)][double]$CloseSeconds = 24.0,
    [switch]$CloseOnStableContact,
    [ValidateRange(-180.0,180.0)][double]$GraspHeadingReferenceDegrees = 0.0,
    [ValidateRange(0.0,10.0)][double]$GraspTiltDegrees = 0.0,
    [ValidateRange(0.01, 300.0)][double]$NavigationBudgetSeconds = 48.0,
    [int]$FleetTicks = 0
)

if ([string]::IsNullOrWhiteSpace($BatchCodes)) {
    $BatchCodes = $(if ($StreamingFeed) { '1-401,23-402,2-403,24-404,3-405,25-406' } else { '1-401,23-402' })
}
if ($FleetTicks -lt 0) { throw 'FleetTicks must be positive or zero for the recipe default.' }
if ($FleetTicks -eq 0) { $FleetTicks = $(if ($StreamingFeed) { 360000 } else { 120000 }) }
$checkpointFile = (Resolve-Path -LiteralPath $Checkpoint -ErrorAction Stop).Path
$recipe = @{
    Headless = $Headless.IsPresent
    RunDirectory = $RunDirectory
    Checkpoint = $checkpointFile
    ArmModel = 'z1'
    Z1CarryLift = $CarryLift
    Z1JawSampleX = $JawSampleX
    Z1ApproachSeconds = $ApproachSeconds
    Z1LiftSeconds = $LiftSeconds
    Z1CloseSeconds = $CloseSeconds
    Z1CloseOnStableContact = $CloseOnStableContact.IsPresent
    Z1GraspTiltDegrees = $GraspTiltDegrees
    Z1DepartureClearance = $DepartureClearance.IsPresent
    Z1SettleStow = $SettleStow.IsPresent
    Z1StowBrakeSupport = $StowBrakeSupport.IsPresent
    Z1LoadedBrakeSupport = $LoadedBrakeSupport.IsPresent
    Z1ReceivingBrakeSupport = $ReceivingBrakeSupport.IsPresent
    Z1PlacementWrenchQP = $PlacementWrenchQP.IsPresent
    Z1LoadedBrakeWrenchQP = $LoadedBrakeWrenchQP.IsPresent
    Z1TotalMotorWrenchQP = $TotalMotorWrenchQP.IsPresent
    Z1JointMotorWrenchQP = $JointMotorWrenchQP.IsPresent
    Z1LoadedWaitWrenchQP = $LoadedWaitWrenchQP.IsPresent
    Z1ManipulationMotorQP = $ManipulationMotorQP.IsPresent
    Z1LoadedSupportHysteresis = $LoadedSupportHysteresis.IsPresent
    TouchdownIK = $AirborneTouchdown.IsPresent
    AirborneTouchdown = $AirborneTouchdown.IsPresent
    Z1ContactWrench = $ContactWrench.IsPresent
    Z1ReturnCheckpoint = $ReturnCheckpoint
    Z1CarryCheckpoint = $CarryCheckpoint
    Z1EmptyTransitSpecialist = $EmptyTransitSpecialist.IsPresent
    Z1ResourceWaitSpecialist = $ResourceWaitSpecialist.IsPresent
    Z1RepeatPickupSpecialist = $RepeatPickupSpecialist.IsPresent
    Z1SpecialistIdleControl = $SpecialistIdleControl.IsPresent
    Seed = $Seed
    StationaryMode = 'Balance'
    BalanceScope = 'Settled'
    MeasuredCOM = $true
    ContinuousRecoveryTargets = $true
    WaypointRamp = $true
    VectorTracking = $true
    LoadedSpeedLimit = 0.35
    ReturnSpeedLimit = $ReturnSpeedLimit
    LoadedRecoveryTriggerHeight = $LoadedRecoveryTriggerHeight
    PlacementPrecision = $true
    SegmentTransit = $true
    PipelineReceiving = $PipelineReceiving.IsPresent
    Z1DockingAxisHysteresis = $DockingAxisHysteresis.IsPresent
    CompactReceiving = $CompactReceiving.IsPresent
    ReceivingApproachX = $ReceivingApproachX
    ReceivingApproachStagedOnly = $ReceivingApproachStagedOnly.IsPresent
    SettleReceiving = $SettleReceiving.IsPresent
    ShortSpineReservation = $ShortSpineReservation.IsPresent
    PlanPeerClearance = $true
    SettleReservation = $SettleReservation.IsPresent
    HandoffReleaseSeconds = $HandoffReleaseSeconds
    StreamingFeed = $StreamingFeed.IsPresent
    DynamicObstacle = $DynamicObstacle.IsPresent
    BatchCodes = $BatchCodes
    FleetTicks = $FleetTicks
    NavigationBudgetSeconds = $NavigationBudgetSeconds
}
if($PSBoundParameters.ContainsKey('GraspHeadingReferenceDegrees')){
    $recipe.Z1GraspHeadingReferenceDegrees=$GraspHeadingReferenceDegrees
}
if($PSBoundParameters.ContainsKey('PregraspSeconds')){
    $recipe.Z1PregraspSeconds=$PregraspSeconds
}
& (Join-Path $PSScriptRoot 'run_warehouse_demo.ps1') @recipe
