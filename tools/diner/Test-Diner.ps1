[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$GameExe,
    [Parameter(Mandatory)][string]$PlayerData,
    [string]$Renderer = (Join-Path $PSScriptRoot '..\..\target\diag\victory.exe'),
    [string]$WarmCache,
    [string]$WorkingDirectory = (Join-Path $PSScriptRoot '..\..'),
    [string]$Reports = (Join-Path $PSScriptRoot 'runs'),
    [ValidateRange(10,600)][int]$Seconds = 240
)
$ErrorActionPreference='Stop'
$game=(Get-Item -LiteralPath $GameExe).FullName
$exe=(Get-Item -LiteralPath $Renderer).FullName
$work=(Get-Item -LiteralPath $WorkingDirectory).FullName
if (!(Test-Path -LiteralPath (Join-Path $PlayerData '7952987A\saves') -PathType Container)) {
    throw 'PlayerData must point to a VIctory data directory containing 7952987A\saves.'
}
if ($WarmCache -and !(Test-Path -LiteralPath (Join-Path $WarmCache 'shaders') -PathType Container)) {
    throw 'WarmCache must point to a cache directory containing shaders.'
}
if (Get-Process -Name ([IO.Path]::GetFileNameWithoutExtension($exe)) -ErrorAction SilentlyContinue) {
    throw 'A renderer with this executable name is already running.'
}
$id=[guid]::NewGuid().ToString('N').Substring(0,8)
$out=(New-Item -ItemType Directory -Force -Path (Join-Path $Reports ((Get-Date -Format yyyyMMdd-HHmmss)+'-'+$id))).FullName
Write-Output "Session directory: $out"
Add-Type -Path (Join-Path $PSScriptRoot 'WorldMemory.cs')
$m=[WorldMemory]::Sample()
if ($m[1]-$m[0] -lt 24.25GB -or $m[2] -lt 2GB) {throw 'System memory headroom is too low for the measured Series X reference.'}
$data=Join-Path $out 'data'
$cache=Join-Path $out 'cache'
$frames=Join-Path $out 'frames'
New-Item -ItemType Directory -Path $frames,(Join-Path $out 'guest-pages') | Out-Null
Copy-Item -LiteralPath $PlayerData -Destination $data -Recurse
if ($WarmCache) {Copy-Item -LiteralPath $WarmCache -Destination $cache -Recurse}
else {New-Item -ItemType Directory -Path $cache | Out-Null}
$prior=@{}
foreach ($v in Get-ChildItem Env:) {
    if ($v.Name -like 'VICTORY_*' -or $v.Name -eq 'RUST_LOG') { $prior[$v.Name]=$v.Value }
}
$settings=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'reference-settings.json') -Raw | ConvertFrom-Json
$p=$null
$writer=$null
$telemetry=$null
$gpuTask=$null
$state=[ordered]@{exe=$exe;sha256=(Get-FileHash $exe).Hash;warmCache=$WarmCache;seconds=$Seconds;outcome='startup';workingSetGiB=16}
try {
    foreach ($name in $prior.Keys) {Remove-Item -LiteralPath ('Env:'+$name)}
    foreach ($prop in $settings.PSObject.Properties) { Set-Item -LiteralPath ('Env:'+$prop.Name) -Value $prop.Value }
    $env:VICTORY_GAME_DIR=Split-Path $game
    $env:VICTORY_CACHE_DIR=$cache
    $env:VICTORY_DATA_DIR=$data
    $env:VICTORY_CAPTURE=$frames
    $env:VICTORY_PHYSICAL_BACKING_DIR=Join-Path $out 'guest-pages'
    $p=Start-Process -FilePath $exe -ArgumentList @('run',('"'+$game+'"'),'--execute') -WorkingDirectory $work -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $out 'stdout.log') -RedirectStandardError (Join-Path $out 'stderr.log')
    [WorldMemory]::LimitResidentMemory($p.Handle,16)
    $state.rendererPid=$p.Id
    $state.outcome='running'
    $state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'run.json')
    $writer=[IO.StreamWriter]::new((Join-Path $out 'memory.csv'))
    $writer.AutoFlush=$true
    $writer.WriteLine('elapsed_ms,private_bytes,working_set_bytes,system_commit_bytes,commit_limit_bytes,available_physical_bytes')
    $timer=[Diagnostics.Stopwatch]::StartNew()
    $telemetry=[PowerShell]::Create()
    [void]$telemetry.AddScript({
        param($rendererId,$path,$start)
        $gpuWriter=[IO.StreamWriter]::new($path)
        $gpuWriter.AutoFlush=$true
        try {
            $gpuWriter.WriteLine('elapsed_ms,instance,dedicated_bytes,shared_bytes,total_committed_bytes')
            while (Get-Process -Id $rendererId -ErrorAction SilentlyContinue) {
                $rows=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -Filter ("Name LIKE 'pid_{0}[_]%'" -f $rendererId)
                foreach ($row in $rows) {
                    $gpuWriter.WriteLine(('{0},{1},{2},{3},{4}' -f [int]([DateTime]::UtcNow-$start).TotalMilliseconds,$row.Name,$row.DedicatedUsage,$row.SharedUsage,$row.TotalCommitted))
                }
                Start-Sleep -Seconds 2
            }
        } finally { $gpuWriter.Dispose() }
    }).AddArgument($p.Id).AddArgument((Join-Path $out 'gpu.csv')).AddArgument([DateTime]::UtcNow)
    $gpuTask=$telemetry.BeginInvoke()
    while (!$p.HasExited) {
        $m=[WorldMemory]::Sample()
        $p.Refresh()
        if ($p.HasExited) {break}
        $writer.WriteLine(('{0},{1},{2},{3},{4},{5}' -f $timer.ElapsedMilliseconds,$p.PrivateMemorySize64,$p.WorkingSet64,$m[0],$m[1],$m[2]))
        if ($m[1]-$m[0] -lt 1.5GB) {$state.outcome='commit-guard';break}
        if ($m[2] -lt 2GB) {$state.outcome='physical-memory-guard';break}
        if (Test-Path -LiteralPath (Join-Path $out 'scene-reviewed.stop')) {$state.outcome='scene-reviewed';break}
        if ($timer.Elapsed.TotalSeconds -ge $Seconds) {$state.outcome='duration';break}
        Start-Sleep -Milliseconds 200
    }
    if ($state.outcome -eq 'running') {$state.outcome='application-exit'}
} catch {
    $state.outcome='error'
    throw
} finally {
    if ($p) {
        if (!$p.HasExited) {$p.Kill()}
        $p.WaitForExit()
        $state.exitCode=$p.ExitCode
        $p.Dispose()
    }
    if ($writer) {$writer.Dispose()}
    if ($telemetry) {
        if ($gpuTask) { [void]$telemetry.EndInvoke($gpuTask) }
        $state.gpuSamplerErrors=@($telemetry.Streams.Error | ForEach-Object { [string]$_ } | Select-Object -Unique)
        $telemetry.Dispose()
    }
    foreach ($v in @(Get-ChildItem Env:)) {
        if ($v.Name -like 'VICTORY_*' -or $v.Name -eq 'RUST_LOG') {Remove-Item -LiteralPath ('Env:'+$v.Name)}
    }
    foreach ($name in $prior.Keys) {Set-Item -LiteralPath ('Env:'+$name) -Value $prior[$name]}
    $state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'run.json')
}
Write-Output "$($state.outcome) | $out"
