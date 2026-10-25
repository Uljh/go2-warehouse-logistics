param(
    [switch]$Headless,
    [string]$RunDirectory = '',
    [switch]$WaitForStart,
    [switch]$NavigationOnly,
    [switch]$NavigationObstacleRoute,
    [int]$Seed = 8,
    [string]$Checkpoint = 'E:/Robot_Project/artifacts/go2_official_checkpoint.pt',
    [ValidateSet('panda_surrogate','z1')][string]$ArmModel = 'panda_surrogate',
    [string]$Z1ReturnCheckpoint = '',
    [ValidateRange(0.44, 0.52)][double]$Z1CarryLift = 0.44,
    [ValidateRange(0.14, 0.16)][double]$Z1JawSampleX = 0.15,
    [ValidateRange(4.0, 8.0)][double]$Z1ApproachSeconds = 8.0,
    [ValidateRange(8.0, 24.0)][double]$Z1CloseSeconds = 24.0,
    [switch]$Z1CloseOnStableContact,
    [ValidateRange(-180.0,180.0)][double]$Z1GraspHeadingReferenceDegrees = 0.0,
    [switch]$Z1DepartureClearance,
    [switch]$Z1SettleStow,
    [switch]$Z1StowBrakeSupport,
    [switch]$Z1LoadedBrakeSupport,
    [switch]$Z1PlacementWrenchQP,
    [switch]$Z1LoadedBrakeWrenchQP,
    [switch]$Z1ContactWrench,
    [ValidateSet('Captured', 'Policy', 'Balance')][string]$StationaryMode = 'Captured',
    [ValidateSet('force_balance', 'policy')][string]$RecoveryController = 'force_balance',
    [ValidateSet('All','Stationary','Settled','SettledAll')][string]$BalanceScope = 'All',
    [ValidateSet('Original', 'Axis')][string]$RouteMode = 'Original',
    [switch]$RecenterFoot,
    [switch]$OffloadFoot,
    [switch]$ArmPreflight,
    [switch]$ArmPathRefinement,
    [switch]$SymmetricGrasp,
    [switch]$ContinuousRecoveryTargets,
    [switch]$TouchdownIK,
    [switch]$WorkspaceReference,
    [switch]$LowStanceWorkspace,
    [switch]$SupportExitRamp,
    [ValidateRange(0.0, 2.0)][double]$HandoffReleaseSeconds = 0.0,
    [switch]$WaypointRamp,
    [switch]$VectorTracking,
    [ValidateRange(0.0, 0.61)][double]$LoadedSpeedLimit = 0.0,
    [ValidateRange(0.0, 0.50)][double]$ReturnSpeedLimit = 0.0,
    [switch]$PlacementPrecision,
    [switch]$PreSupportYawBrake,
    [ValidateSet('Ideal','Perturbed')][string]$RangeProfile = 'Ideal',
    [ValidateSet('Truth','Perturbed')][string]$LocalizationProfile = 'Truth',
    [ValidateSet(1,2)][int]$PhysicsSubsteps = 1,
    [switch]$ReserveTransit,
    [switch]$SegmentTransit,
    [switch]$PipelineReceiving,
    [switch]$ShortSpineReservation,
    [switch]$PlanPeerClearance,
    [switch]$SettleReservation,
    [switch]$RecoverDrops,
    [switch]$InjectDrop,
    [switch]$RightFallenParcel,
    [switch]$StreamingFeed,
    [switch]$MeasuredCOM,
    [switch]$StanceFit,
    [switch]$TravelHeading,
    [ValidateRange(0.0, 0.20)][double]$LegTargetMargin = 0.0,
    [ValidateSet('Clipped', 'Bounded')][string]$LegIKMode = 'Clipped',
    [ValidateRange(0.0, 0.15)][double]$CarryRetraction = 0.0,
    [ValidateRange(0.08, 0.18)][double]$ReleaseRetreatHeight = 0.08,
    [ValidateRange(0.31, 0.33)][double]$RecoveryHeight = 0.33,
    [ValidateRange(0.285, 0.305)][double]$LoadedRecoveryTriggerHeight = 0.285,
    [string]$BatchCodes = '1-001,23-002',
    [int]$FleetTicks = 30000,
    [int]$MaxControlSteps = 2400,
    [ValidateRange(0.01, 300.0)][double]$NavigationBudgetSeconds = 48.0
)

if ($PSBoundParameters.ContainsKey('MaxControlSteps') -and $PSBoundParameters.ContainsKey('NavigationBudgetSeconds')) {
    throw 'Choose either explicit MaxControlSteps or NavigationBudgetSeconds.'
}

if ($Headless -and $WaitForStart) {
    throw 'WaitForStart requires the visual window. Omit it for headless recording.'
}
if ($NavigationOnly -and ($RecoverDrops -or $InjectDrop)) {
    throw 'NavigationOnly cannot exercise parcel drop recovery.'
}
if ($RightFallenParcel -and -not ($RecoverDrops -or $InjectDrop)) {
    throw 'RightFallenParcel requires RecoverDrops or InjectDrop.'
}
if ($NavigationObstacleRoute -and -not $NavigationOnly) {
    throw 'NavigationObstacleRoute requires NavigationOnly.'
}
if ($OffloadFoot -and -not $RecenterFoot) {
    throw 'OffloadFoot requires RecenterFoot. Both remain opt-in physical experiments.'
}
if ($StanceFit -and $RecenterFoot) {
    throw 'StanceFit and RecenterFoot are separate physical experiments.'
}
if ($ArmPathRefinement -and -not $ArmPreflight) {
    throw 'ArmPathRefinement requires ArmPreflight.'
}
if ($SymmetricGrasp -and -not $ArmPreflight) {
    throw 'SymmetricGrasp requires ArmPreflight.'
}
if ($StationaryMode -eq 'Balance' -and $RecoveryController -ne 'force_balance') {
    throw 'Balance stationary support requires the force_balance controller.'
}
if ($ContinuousRecoveryTargets -and $RecoveryController -ne 'force_balance') {
    throw 'ContinuousRecoveryTargets requires the force_balance recovery controller.'
}
if ($TouchdownIK -and ($RecoveryController -ne 'force_balance' -or $RecenterFoot -or $StanceFit)) {
    throw 'TouchdownIK requires force_balance and is separate from RecenterFoot/StanceFit.'
}
if (($WorkspaceReference -or $LowStanceWorkspace) -and ($RecoveryController -ne 'force_balance' -or $RecenterFoot -or $StanceFit -or $TouchdownIK)) {
    throw 'WorkspaceReference requires force_balance and a separate stance experiment.'
}
if ($BalanceScope -in @('Stationary','Settled','SettledAll') -and ($StationaryMode -ne 'Balance' -or $RecoveryController -ne 'force_balance')) {
    throw 'Stationary BalanceScope requires Balance parking and force_balance support.'
}
if ($ReserveTransit -and $SegmentTransit) {
    throw 'ReserveTransit and SegmentTransit are separate controlled transport experiments.'
}
if ($ShortSpineReservation -and -not $SegmentTransit) {
    throw 'ShortSpineReservation requires SegmentTransit.'
}
if ($Z1LoadedBrakeWrenchQP -and -not ($ArmModel -eq 'z1' -and $Z1LoadedBrakeSupport)) {
    throw 'Z1LoadedBrakeWrenchQP requires original Z1 and Z1LoadedBrakeSupport.'
}
if ($PlanPeerClearance -and -not ($ReserveTransit -or $SegmentTransit)) {
    throw 'PlanPeerClearance requires ReserveTransit or SegmentTransit.'
}
if ($StreamingFeed -and $NavigationOnly) {
    throw 'StreamingFeed requires a delivery mission (omit NavigationOnly).'
}
if ($StreamingFeed -and (($BatchCodes -split ',').Count -lt 6 -or ($BatchCodes -split ',').Count -gt 10)) {
    throw 'StreamingFeed requires 6..10 comma-separated unique batch codes.'
}
if ($PreSupportYawBrake -and $BalanceScope -notin @('Settled','SettledAll')) {
    throw 'PreSupportYawBrake requires a measured settled support scope.'
}

# A self-contained candidate recipe: unrelated previous experiments must not
# silently select another allocator or change recovery inputs.
$warehouseRecipe = @{
    GO2_Z1_RETURN_CHECKPOINT = $(if ($Z1ReturnCheckpoint) { (Resolve-Path -LiteralPath $Z1ReturnCheckpoint -ErrorAction Stop).Path } else { '' })
    GO2_FLEET_NAVIGATION_BUDGET_S = $(if ($PSBoundParameters.ContainsKey('MaxControlSteps')) { '' } else { $NavigationBudgetSeconds.ToString([Globalization.CultureInfo]::InvariantCulture) })
    GO2_Z1_CARRY_LIFT = $Z1CarryLift.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_Z1_JAW_SAMPLE_X = $Z1JawSampleX.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_Z1_APPROACH_SECONDS = $Z1ApproachSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_Z1_CLOSE_SECONDS = $Z1CloseSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_Z1_CLOSE_ON_STABLE_CONTACT = [string][int]$Z1CloseOnStableContact.IsPresent
    GO2_Z1_GRASP_HEADING_REFERENCE_DEG = $(if($PSBoundParameters.ContainsKey('Z1GraspHeadingReferenceDegrees')){
        $Z1GraspHeadingReferenceDegrees.ToString([Globalization.CultureInfo]::InvariantCulture)}else{''})
    GO2_Z1_DEPARTURE_CLEARANCE = [string][int]$Z1DepartureClearance.IsPresent
    GO2_Z1_SETTLE_STOW = [string][int]$Z1SettleStow.IsPresent
    GO2_Z1_STOW_BRAKE_SUPPORT = [string][int]$Z1StowBrakeSupport.IsPresent
    GO2_Z1_LOADED_BRAKE_SUPPORT = [string][int]$Z1LoadedBrakeSupport.IsPresent
    GO2_Z1_PLACEMENT_WRENCH_QP = [string][int]$Z1PlacementWrenchQP.IsPresent
    GO2_Z1_LOADED_BRAKE_WRENCH_QP = [string][int]$Z1LoadedBrakeWrenchQP.IsPresent
    GO2_Z1_CONTACT_WRENCH = [string][int]$Z1ContactWrench.IsPresent
    GO2_FLEET_SETTLE_RESERVATION = [string][int]$SettleReservation.IsPresent
    GO2_BALANCE_ALLOCATOR = 'sequential'
    GO2_BALANCE_HANDOFF_RELEASE_S = $HandoffReleaseSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_BALANCE_SCOPE = $BalanceScope.ToLowerInvariant()
    GO2_FLEET_RECOVERY_FORWARD = '0'
    GO2_FLEET_VELOCITY_SLEW = '0'
    GO2_FLEET_LOADED_SPEED_LIMIT = $LoadedSpeedLimit.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_FLEET_RETURN_SPEED_LIMIT = $ReturnSpeedLimit.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_FLEET_PLACEMENT_PRECISION = [string][int]$PlacementPrecision.IsPresent
    GO2_FLEET_WAYPOINT_RAMP = [string][int]$WaypointRamp.IsPresent
    GO2_FLEET_VECTOR_TRACKING = [string][int]$VectorTracking.IsPresent
    GO2_FLEET_SUPPORT_EXIT_RAMP = [string][int]$SupportExitRamp.IsPresent
    GO2_FLEET_PRE_SUPPORT_YAW_BRAKE = [string][int]$PreSupportYawBrake.IsPresent
    GO2_FLEET_RANGE_PROFILE = $RangeProfile.ToLowerInvariant()
    GO2_FLEET_CONTACT_RIGHTING = [string][int]$RightFallenParcel.IsPresent
    GO2_FLEET_LOCALIZATION_PROFILE = $LocalizationProfile.ToLowerInvariant()
    GO2_FLEET_SMOKE_OBSTACLE_ROUTE = [string][int]$NavigationObstacleRoute.IsPresent
    GO2_PHYSICS_SUBSTEPS = [string]$PhysicsSubsteps
    GO2_FLEET_SETTLE_TRANSITIONS = '0'
    GO2_FLEET_STATIONARY_MODE = $StationaryMode.ToLowerInvariant()
    GO2_FLEET_FORWARD_FLOOR = '0.30'
    GO2_FLEET_AXIS_ROUTES = $(if ($RouteMode -eq 'Axis') { '1' } else { '0' })
    GO2_BALANCE_RECENTER = [string][int]$RecenterFoot.IsPresent
    GO2_BALANCE_OFFLOAD = [string][int]$OffloadFoot.IsPresent
    GO2_FLEET_ARM_PREFLIGHT = [string][int]$ArmPreflight.IsPresent
    GO2_FLEET_ARM_REFINEMENT = [string][int]$ArmPathRefinement.IsPresent
    GO2_FLEET_GRASP_SYMMETRY = [string][int]$SymmetricGrasp.IsPresent
    GO2_BALANCE_MEASURED_COM = [string][int]$MeasuredCOM.IsPresent
    GO2_BALANCE_STANCE_FIT = [string][int]$StanceFit.IsPresent
    GO2_FLEET_TRAVEL_HEADING = [string][int]$TravelHeading.IsPresent
    GO2_FLEET_LEG_TARGET_MARGIN = $LegTargetMargin.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_BALANCE_LEG_IK = $LegIKMode.ToLowerInvariant()
    GO2_BALANCE_TARGET_UPDATE = $(if ($ContinuousRecoveryTargets) { 'project_then_slew' } else { 'legacy' })
    GO2_BALANCE_TOUCHDOWN_IK = [string][int]$TouchdownIK.IsPresent
    GO2_BALANCE_WORKSPACE_REFERENCE = [string][int]($WorkspaceReference.IsPresent -or $LowStanceWorkspace.IsPresent)
    GO2_BALANCE_WORKSPACE_MAX_HEIGHT = $(if ($LowStanceWorkspace) { '0.305' } else { '' })
    GO2_FLEET_RESERVE_TRANSIT = [string][int]$ReserveTransit.IsPresent
    GO2_FLEET_SEGMENT_TRANSIT = [string][int]$SegmentTransit.IsPresent
    GO2_FLEET_PIPELINE_RECEIVING = [string][int]$PipelineReceiving.IsPresent
    GO2_FLEET_SHORT_SPINE_RESERVATION = [string][int]$ShortSpineReservation.IsPresent
    GO2_FLEET_PLAN_PEER_CLEARANCE = [string][int]$PlanPeerClearance.IsPresent
    GO2_FLEET_CARRY_RETRACTION = $CarryRetraction.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_FLEET_RELEASE_RETREAT = $ReleaseRetreatHeight.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_BALANCE_HEIGHT = $RecoveryHeight.ToString([Globalization.CultureInfo]::InvariantCulture)
    GO2_FLEET_LOADED_RECOVERY_TRIGGER_HEIGHT = $LoadedRecoveryTriggerHeight.ToString([Globalization.CultureInfo]::InvariantCulture)
}
$previousWarehouseRecipe = @{}
foreach ($optionName in $warehouseRecipe.Keys) {
    $previousWarehouseRecipe[$optionName] = [Environment]::GetEnvironmentVariable($optionName, 'Process')
    [Environment]::SetEnvironmentVariable($optionName, $warehouseRecipe[$optionName], 'Process')
}
try {
    $warehouseArguments = @{
        Fleet = 2
        ArmModel = $ArmModel
        GraspMode = $(if ($ArmModel -eq 'z1') { 'contact' } else { 'assisted' })
        RunDirectory = $RunDirectory
        FleetSmoke = $NavigationOnly.IsPresent
        IndoorWarehouse = $true
        FleetRecoveryController = $RecoveryController
        FleetRecoveryHandoff = 'settled_slew'
        FleetLateralStall = $true
        FleetDetailVideo = $true
        FleetTicks = $FleetTicks
        MaxControlSteps = $MaxControlSteps
        Seed = $Seed
        Checkpoint = $Checkpoint
        BatchCodes = $BatchCodes
        StreamingFeed = $StreamingFeed.IsPresent
        RecoverDrops = $RecoverDrops.IsPresent
        InjectDrop = $InjectDrop.IsPresent
        Visual = -not $Headless
        WaitForStart = $WaitForStart.IsPresent
    }
    & (Join-Path $PSScriptRoot 'run_go2_mission.ps1') @warehouseArguments
}
finally {
    foreach ($optionName in $previousWarehouseRecipe.Keys) {
        [Environment]::SetEnvironmentVariable($optionName, $previousWarehouseRecipe[$optionName], 'Process')
    }
}
