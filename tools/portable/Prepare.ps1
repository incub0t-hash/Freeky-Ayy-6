[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$GameExe,
    [string]$StateDirectory = (Join-Path $env:LOCALAPPDATA 'Freeky-Ayy-6\best-run')
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot
$game=(Get-Item -LiteralPath $GameExe).FullName
$gameDir=Split-Path $game
if (!(Test-Path -LiteralPath (Join-Path $gameDir 'shaders.rpf'))) {throw 'The game directory must contain shaders.rpf.'}
$state=[IO.Path]::GetFullPath($StateDirectory)
$cache=Join-Path $state 'cache'
$exe=Join-Path $root 'bin\victory.exe'
$compiler=Join-Path $root 'bin\victory-prepare-cache.exe'
$manifest=Get-Content -LiteralPath (Join-Path $root 'package.json') -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $exe).Hash -ne $manifest.rendererSHA256) {throw 'Renderer checksum differs from package.json.'}
& $exe --version
if($LASTEXITCODE){throw 'Install the Microsoft x64 C++ runtime with Install-Runtime.cmd, then retry.'}
New-Item -ItemType Directory -Force -Path $state | Out-Null
$ready=Join-Path $state 'prepared.json'
$gameHash=(Get-FileHash -LiteralPath $game).Hash
$hostConfiguration=(& $compiler probe | Out-String).Trim()
if($LASTEXITCODE){throw 'The graphics adapter could not be prepared for Direct3D 12.'}
if(Test-Path -LiteralPath $ready) {
    $prior=Get-Content -LiteralPath $ready -Raw|ConvertFrom-Json
    if($prior.rendererSHA256 -eq $manifest.rendererSHA256 -and $prior.gameSHA256 -eq $gameHash -and $prior.hostConfiguration -eq $hostConfiguration -and
       (Test-Path -LiteralPath (Join-Path $cache 'shaders'))) {
        Write-Output "Prepared cache is available in $cache"
        return
    }
    if($prior.hostConfiguration -ne $hostConfiguration -and (Test-Path -LiteralPath (Join-Path $cache 'shaders'))) {
        $old=Join-Path $cache 'shaders'
        $backup=Join-Path $cache ('shaders-'+[guid]::NewGuid().ToString('N'))
        foreach($path in @($old,$backup)) {
            if(!([IO.Path]::GetFullPath($path).StartsWith([IO.Path]::GetFullPath($cache)+'\',[StringComparison]::OrdinalIgnoreCase))) {
                throw 'Unexpected cache backup path.'
            }
        }
        [IO.Directory]::Move($old,$backup)
    }
}
$before=@{}
foreach($v in @(Get-ChildItem Env:|Where-Object Name -Like 'VICTORY_*')) {
    $before[$v.Name]=$v.Value
    Remove-Item -LiteralPath ('Env:'+$v.Name)
}
try {
    $arguments=@('seed',$exe,(Join-Path $cache 'shaders'),(Join-Path $PSScriptRoot 'shader-witness.tsv'),(Join-Path $gameDir 'shaders.rpf'),$game) |
        ForEach-Object { '"'+$_+'"' }
    $process=Start-Process -FilePath $compiler -ArgumentList $arguments -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $state 'prepare.log') -RedirectStandardError (Join-Path $state 'prepare-errors.log')
    if($process.ExitCode){throw "Cache preparation failed. See $state\prepare-errors.log"}
    $log=Get-Content -LiteralPath (Join-Path $state 'prepare.log') -Raw
    if($log -notmatch 'found=2492 verified=2492 hits=') {throw 'The supplied game did not reproduce the 2492 reference shader inputs.'}
    @{rendererSHA256=$manifest.rendererSHA256;gameSHA256=$gameHash;gameExe=$game;cache=$cache;hostConfiguration=$hostConfiguration;verifiedShaders=2492} |
        ConvertTo-Json | Set-Content -LiteralPath $ready
} finally {
    foreach($name in $before.Keys){Set-Item -LiteralPath ('Env:'+$name) -Value $before[$name]}
}
Write-Output "Prepared cache from your game files in $cache"
