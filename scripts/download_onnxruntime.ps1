$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Version = "1.23.2"
$DownloadUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-$Version.zip"
$ThirdPartyDir = Join-Path $ProjectRoot "third_party"
$ArchivePath = Join-Path $ThirdPartyDir "onnxruntime-win-x64-$Version.zip"
$ExtractedDir = Join-Path $ThirdPartyDir "onnxruntime-win-x64-$Version"
$TargetDir = Join-Path $ThirdPartyDir "onnxruntime"

New-Item -ItemType Directory -Force -Path $ThirdPartyDir | Out-Null

if (-not (Test-Path -LiteralPath $ArchivePath)) {
    Write-Host "Downloading ONNX Runtime $Version..."
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ArchivePath
}

if (Test-Path -LiteralPath $ExtractedDir) {
    Remove-Item -LiteralPath $ExtractedDir -Recurse -Force
}

Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ThirdPartyDir -Force

if (Test-Path -LiteralPath $TargetDir) {
    Remove-Item -LiteralPath $TargetDir -Recurse -Force
}

Move-Item -LiteralPath $ExtractedDir -Destination $TargetDir
Write-Host "ONNX Runtime installed to: $TargetDir"
