param(
    [string]$OpenCvBin = $env:OpenCV_BIN
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ExePath = Join-Path $ProjectRoot "build\Release\industrial_vision_defect.exe"

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Executable not found. Run scripts/build_release.ps1 first."
}

if ($OpenCvBin) {
    $env:PATH = "$OpenCvBin;$env:PATH"
}

Push-Location $ProjectRoot
try {
    & $ExePath --demo --show 0 --save 1
} finally {
    Pop-Location
}
