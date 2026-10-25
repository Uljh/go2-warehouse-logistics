# Bounded six-parcel queue. Every parcel exists before simulation starts.
param(
    [switch]$Headless,
    [switch]$SegmentTransit,
    [switch]$TouchdownIK,
    [switch]$WorkspaceReference,
    [switch]$LowStanceWorkspace,
    [string]$RunDirectory = '',
    [int]$Seed = 8,
    [string]$BatchCodes = '1-101,23-102,2-103,24-104,3-105,25-106',
    [int]$FleetTicks = 60000
)

& (Join-Path $PSScriptRoot 'run_localized_demo.ps1') -Headless:$Headless `
    -SegmentTransit:$SegmentTransit -StreamingFeed -Seed $Seed `
    -BatchCodes $BatchCodes -FleetTicks $FleetTicks -RunDirectory $RunDirectory `
    -TouchdownIK:$TouchdownIK -WorkspaceReference:$WorkspaceReference -LowStanceWorkspace:$LowStanceWorkspace
