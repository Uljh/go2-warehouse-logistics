function Assert-FleetMissionResult {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ResultPath,
        [Parameter(Mandatory)][string]$LaunchToken,
        [switch]$RequireGuiCheck
    )
    if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
        throw 'Fleet produced no result for this launch; process exit code alone cannot confirm success.'
    }
    $result = Get-Content -LiteralPath $ResultPath -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    if ($result.launch_token -ne $LaunchToken) {
        throw 'Fleet result belongs to another launch or is stale; this run has no verified result.'
    }
    if ($result.state -ne 'complete') {
        throw "Fleet mission incomplete: $($result.failure). Evidence: $($result.run_dir)"
    }
    if ($RequireGuiCheck) {
        $guiPath = Join-Path $result.run_dir 'gui_check.json'
        if (-not (Test-Path -LiteralPath $guiPath -PathType Leaf)) {
            throw 'Fleet GUI verification has no completed report.'
        }
        $gui = Get-Content -LiteralPath $guiPath -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        if ($gui.passed -ne $true) {
            throw "Fleet GUI verification failed: $guiPath"
        }
    }
}
