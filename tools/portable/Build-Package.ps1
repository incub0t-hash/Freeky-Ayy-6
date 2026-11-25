[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Destination,
    [Parameter(Mandatory)][string]$Renderer,
    [Parameter(Mandatory)][string]$CheckpointData,
    [string]$CachePreparer = (Join-Path $PSScriptRoot '..\..\target\diag\victory-prepare-cache.exe'),
    [Parameter(Mandatory)][string]$RuntimeInstaller,
    [string]$SourceChanges
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$out=[IO.Path]::GetFullPath($Destination)
if(Test-Path -LiteralPath $out){throw 'Choose a new package destination.'}
foreach($path in @($Renderer,$CachePreparer,$RuntimeInstaller)){if(!(Test-Path -LiteralPath $path -PathType Leaf)){throw "Missing $path"}}
if(!(Test-Path -LiteralPath (Join-Path $CheckpointData '7952987A\saves'))){throw 'CheckpointData must contain 7952987A\saves.'}
$signature=Get-AuthenticodeSignature -LiteralPath $RuntimeInstaller
if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation'){throw 'The runtime installer needs a valid Microsoft signature.'}
New-Item -ItemType Directory -Path $out,(Join-Path $out 'bin'),(Join-Path $out 'scripts'),(Join-Path $out 'seed\data\7952987A'),(Join-Path $out 'prerequisites')|Out-Null
Copy-Item -LiteralPath $Renderer -Destination (Join-Path $out 'bin\victory.exe')
Copy-Item -LiteralPath $CachePreparer -Destination (Join-Path $out 'bin\victory-prepare-cache.exe')
Copy-Item -LiteralPath $RuntimeInstaller -Destination (Join-Path $out 'prerequisites\vc_redist.x64.exe')
foreach($name in @('Play.ps1','Prepare.ps1','Verify-Package.ps1','shader-witness.tsv')){
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $out 'scripts')
}
foreach($name in @('Test-Diner.ps1','WorldMemory.cs','reference-settings.json')){
    Copy-Item -LiteralPath (Join-Path $repo "tools\diner\$name") -Destination (Join-Path $out 'scripts')
}
foreach($name in @('Play.cmd','Install-Runtime.cmd','README.md')){
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $out
}
foreach($name in @('saves','storage')){
    $source=Join-Path $CheckpointData "7952987A\$name"
    if(Test-Path -LiteralPath $source){Copy-Item -LiteralPath $source -Destination (Join-Path $out 'seed\data\7952987A') -Recurse}
}
foreach($name in @('LICENSE','UPSTREAM_README.md')){
    Copy-Item -LiteralPath (Join-Path $repo $name) -Destination $out
}
if($SourceChanges){Copy-Item -LiteralPath $SourceChanges -Destination (Join-Path $out 'local-source-changes') -Recurse}
& git -C $repo bundle create (Join-Path $out 'source.bundle') --branches HEAD
if($LASTEXITCODE){throw 'Source bundle creation failed.'}
$files=@(Get-ChildItem -LiteralPath $out -Recurse -File|ForEach-Object {
    @{path=$_.FullName.Substring($out.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash;bytes=$_.Length}
})
@{rendererRevision='9e05a6eb2a49ceb14370b598c5dfc215ffe03a25';sourceRevision=(& git -C $repo rev-parse HEAD);rendererSHA256=(Get-FileHash -LiteralPath $Renderer).Hash;files=$files} |
    ConvertTo-Json -Depth 5 |Set-Content -LiteralPath (Join-Path $out 'package.json')
Write-Output "Package prepared in $out"
