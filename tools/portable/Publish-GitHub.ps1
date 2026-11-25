[CmdletBinding()]
param([Parameter(Mandatory)][string]$PackageZip)
$ErrorActionPreference='Stop'
$repo='incub0t-hash/Freeky-Ayy-6'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$zip=(Get-Item -LiteralPath $PackageZip).FullName
$gh=Get-Command gh -ErrorAction SilentlyContinue
if(!$gh){throw 'Install GitHub CLI with winget install --id GitHub.cli --exact, then run gh auth login.'}
& git -C $root push publication diner-reference:main
if($LASTEXITCODE){throw 'Main branch upload failed.'}
& git -C $root push publication 'refs/heads/research/*:refs/heads/research/*'
if($LASTEXITCODE){throw 'Development branch upload failed.'}
$tag='best-run-20261124'
& $gh.Source release view $tag --repo $repo *> $null
if($LASTEXITCODE) {
    & $gh.Source release create $tag $zip --repo $repo --target main --title 'Best rendered diner reference and portable runtime' --notes-file (Join-Path $root 'docs\portable-release.md')
} else {
    & $gh.Source release upload $tag $zip --repo $repo
}
if($LASTEXITCODE){throw 'Portable release upload failed.'}
