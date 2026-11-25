$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot
$manifest=Get-Content -LiteralPath (Join-Path $root 'package.json') -Raw |ConvertFrom-Json
foreach($file in $manifest.files) {
    $path=Join-Path $root $file.path
    if (!(Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-FileHash -LiteralPath $path).Hash -ne $file.sha256) {throw "Package checksum mismatch: $($file.path)"}
}
Write-Output "Verified $($manifest.files.Count) packaged files."
