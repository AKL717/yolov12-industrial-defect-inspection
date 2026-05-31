param(
    [string]$OpenCvBin = $env:OpenCV_BIN
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ExePath = Join-Path $ProjectRoot "build\Release\industrial_vision_defect.exe"
$DatasetImages = Join-Path $ProjectRoot "data\datasets\NEU-DET\valid\images"
$OutputDir = Join-Path $ProjectRoot "results_neu_eval"

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Executable not found. Run scripts/build_release.ps1 first."
}

if (-not (Test-Path -LiteralPath $DatasetImages)) {
    throw "NEU-DET validation images not found: $DatasetImages"
}

if ($OpenCvBin) {
    $env:PATH = "$OpenCvBin;$env:PATH"
}

Push-Location $ProjectRoot
try {
    & $ExePath --input $DatasetImages --output $OutputDir --show 0 --save 1 --evaluate 1 --iou 0.3
} finally {
    Pop-Location
}
