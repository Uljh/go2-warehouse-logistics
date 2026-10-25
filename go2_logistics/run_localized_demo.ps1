# Reproduce the seed-8, 290.94-second physical validation from 2026-10-20.
# This is a bounded synthetic-localization candidate, not a SLAM deployment.
param(
    [switch]$Headless,
    [string]$RunDirectory = '',
    [int]$Seed = 8,
    [switch]$RecoverDrops,
    [switch]$InjectDrop,
    [switch]$RightFallenParcel,
    [switch]$StreamingFeed,
    [switch]$SegmentTransit,
    [switch]$TouchdownIK,
    [switch]$WorkspaceReference,
    [switch]$LowStanceWorkspace,
    [string]$BatchCodes = '1-001,23-002',
    [int]$FleetTicks = 30000
)

$localizedDemo = @{
    Headless = $Headless.IsPresent
    RunDirectory = $RunDirectory
    Seed = $Seed
    RouteMode = 'Axis'
    StationaryMode = 'Balance'
    BalanceScope = 'All'
    ArmPreflight = $true
    ArmPathRefinement = $true
    SymmetricGrasp = $true
    CarryRetraction = 0.10
    ReleaseRetreatHeight = 0.16
    ReserveTransit = -not $SegmentTransit.IsPresent
    SegmentTransit = $SegmentTransit.IsPresent
    TouchdownIK = $TouchdownIK.IsPresent
    WorkspaceReference = $WorkspaceReference.IsPresent
    LowStanceWorkspace = $LowStanceWorkspace.IsPresent
    StreamingFeed = $StreamingFeed.IsPresent
    BatchCodes = $BatchCodes
    FleetTicks = $FleetTicks
    PlanPeerClearance = $true
    WaypointRamp = $true
    RangeProfile = 'Perturbed'
    LocalizationProfile = 'Perturbed'
    RecoverDrops = $RecoverDrops.IsPresent
    InjectDrop = $InjectDrop.IsPresent
    RightFallenParcel = $RightFallenParcel.IsPresent
}
& (Join-Path $PSScriptRoot 'run_warehouse_demo.ps1') @localizedDemo
