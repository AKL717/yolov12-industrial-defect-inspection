$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ExePath = Join-Path $ProjectRoot "build_onnx\Release\industrial_vision_defect.exe"
$DefaultInput = Join-Path $ProjectRoot "data\datasets\NEU-DET\valid\images"
$DefaultModel = Join-Path $ProjectRoot "models\neu_yolo12n.onnx"
$DefaultOutput = Join-Path $ProjectRoot "results_dataset_batch"

$InputDir = if ($args.Count -ge 1) { $args[0] } else { $DefaultInput }
$OutputDir = if ($args.Count -ge 2) { $args[1] } else { $DefaultOutput }
$ModelPath = if ($args.Count -ge 3) { $args[2] } else { $DefaultModel }

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Executable not found. Run scripts/build_onnx_release.ps1 first."
}
if (-not (Test-Path -LiteralPath $InputDir)) {
    throw "Input directory not found: $InputDir"
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "ONNX model not found: $ModelPath"
}

Push-Location $ProjectRoot
try {
    & $ExePath --backend onnx --model $ModelPath --input-size 640 --input $InputDir --output $OutputDir --show 0 --save 1 --evaluate 1
} finally {
    Pop-Location
}
