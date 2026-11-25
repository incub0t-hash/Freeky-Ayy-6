[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$GameExe,
    [string]$StateDirectory = (Join-Path $env:LOCALAPPDATA 'Freeky-Ayy-6\best-run'),
    [string]$PlayerData,
    [ValidateRange(10,1200)][int]$Seconds = 240,
    [switch]$PlaySession
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot
& (Join-Path $PSScriptRoot 'Verify-Package.ps1')
& (Join-Path $PSScriptRoot 'Prepare.ps1') -GameExe $GameExe -StateDirectory $StateDirectory
if(!$PlayerData){$PlayerData=Join-Path $root 'seed\data'}
$arguments=@{
    GameExe=$GameExe
    PlayerData=$PlayerData
    Renderer=(Join-Path $root 'bin\victory.exe')
    WarmCache=(Join-Path $StateDirectory 'cache')
    WorkingDirectory=$root
    Reports=(Join-Path $StateDirectory 'sessions')
    Seconds=$Seconds
    PlaySession=$PlaySession
}
& (Join-Path $PSScriptRoot 'Test-Diner.ps1') @arguments
