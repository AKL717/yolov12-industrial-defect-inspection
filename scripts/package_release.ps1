$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OnnxReleaseExeDir = Join-Path $ProjectRoot "build_onnx\Release"
$DefaultReleaseExeDir = Join-Path $ProjectRoot "build\Release"
$ReleaseExeDir = $DefaultReleaseExeDir
if (Test-Path -LiteralPath (Join-Path $OnnxReleaseExeDir "industrial_vision_defect.exe")) {
    $ReleaseExeDir = $OnnxReleaseExeDir
}
$PackageRoot = Join-Path $ProjectRoot "dist\IndustrialVisionDefectSystem"
$ZipPath = Join-Path $ProjectRoot "dist\IndustrialVisionDefectSystem_release.zip"

if (-not (Test-Path -LiteralPath (Join-Path $ReleaseExeDir "industrial_vision_defect.exe"))) {
    throw "Executable not found. Run scripts/build_release.ps1 first."
}

if (Test-Path -LiteralPath $PackageRoot) {
    Remove-Item -LiteralPath $PackageRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $PackageRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "bin") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "configs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "docs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "scripts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "sample_data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "source") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "models") | Out-Null

Copy-Item -LiteralPath (Join-Path $ReleaseExeDir "industrial_vision_defect.exe") -Destination (Join-Path $PackageRoot "bin") -Force
Copy-Item -LiteralPath (Join-Path $ReleaseExeDir "opencv_world430.dll") -Destination (Join-Path $PackageRoot "bin") -Force
if (Test-Path -LiteralPath (Join-Path $ReleaseExeDir "onnxruntime.dll")) {
    Copy-Item -LiteralPath (Join-Path $ReleaseExeDir "onnxruntime.dll") -Destination (Join-Path $PackageRoot "bin") -Force
}
Copy-Item -LiteralPath (Join-Path $ProjectRoot "configs\config.yaml") -Destination (Join-Path $PackageRoot "configs") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "README.md") -Destination $PackageRoot -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\resume_project.md") -Destination (Join-Path $PackageRoot "docs") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\run_demo.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\build_release.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\build_onnx_release.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\download_onnxruntime.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\evaluate_neu_valid.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\setup_yolo_env.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\run_dataset_batch_onnx.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\run_dataset_realtime_onnx.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\train_yolo12_neu.py") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\export_yolo12_onnx.py") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\train_yolo12_neu_edith.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\export_yolo12_onnx_edith.ps1") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\polars.py") -Destination (Join-Path $PackageRoot "scripts") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "requirements-yolo.txt") -Destination $PackageRoot -Force

$SmokeModel = Join-Path $ProjectRoot "models\neu_yolo12n_smoke.onnx"
if (Test-Path -LiteralPath $SmokeModel) {
    Copy-Item -LiteralPath $SmokeModel -Destination (Join-Path $PackageRoot "models") -Force
}
$FinalModel = Join-Path $ProjectRoot "models\neu_yolo12n.onnx"
if (Test-Path -LiteralPath $FinalModel) {
    Copy-Item -LiteralPath $FinalModel -Destination (Join-Path $PackageRoot "models") -Force
}

Copy-Item -LiteralPath (Join-Path $ProjectRoot "CMakeLists.txt") -Destination (Join-Path $PackageRoot "source") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "include") -Destination (Join-Path $PackageRoot "source\include") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "src") -Destination (Join-Path $PackageRoot "source\src") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "configs") -Destination (Join-Path $PackageRoot "source\configs") -Recurse -Force

$SampleImage = Join-Path $ProjectRoot "data\datasets\NEU-DET\valid\images\crazing_1.jpg"
$SampleLabel = Join-Path $ProjectRoot "data\datasets\NEU-DET\valid\labels\crazing_1.txt"
if (Test-Path -LiteralPath $SampleImage) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "sample_data\images") | Out-Null
    Copy-Item -LiteralPath $SampleImage -Destination (Join-Path $PackageRoot "sample_data\images") -Force
}
if (Test-Path -LiteralPath $SampleLabel) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "sample_data\labels") | Out-Null
    Copy-Item -LiteralPath $SampleLabel -Destination (Join-Path $PackageRoot "sample_data\labels") -Force
}

$RunBat = Join-Path $PackageRoot "run_sample.bat"
Set-Content -LiteralPath $RunBat -Encoding ASCII -Value @(
    "@echo off",
    "cd /d %~dp0",
    "bin\industrial_vision_defect.exe --input sample_data\images --output results_sample --show 0 --save 1 --evaluate 1",
    "echo.",
    "echo To run ONNX YOLO after exporting a model:",
    "echo bin\industrial_vision_defect.exe --backend onnx --model models\neu_yolo12n.onnx --input sample_data\images --show 0 --save 1",
    "echo.",
    "echo Results saved to results_sample",
    "pause"
)

$RunOnnxBat = Join-Path $PackageRoot "run_onnx_sample.bat"
Set-Content -LiteralPath $RunOnnxBat -Encoding ASCII -Value @(
    "@echo off",
    "cd /d %~dp0",
    "if not exist models\neu_yolo12n_smoke.onnx (",
    "  echo Missing models\neu_yolo12n_smoke.onnx",
    "  pause",
    "  exit /b 1",
    ")",
    "bin\industrial_vision_defect.exe --backend onnx --model models\neu_yolo12n_smoke.onnx --input-size 320 --input sample_data\images --output results_onnx_sample --show 0 --save 1 --evaluate 1",
    "echo.",
    "echo ONNX results saved to results_onnx_sample",
    "pause"
)

$RunFinalOnnxBat = Join-Path $PackageRoot "run_final_onnx_sample.bat"
Set-Content -LiteralPath $RunFinalOnnxBat -Encoding ASCII -Value @(
    "@echo off",
    "cd /d %~dp0",
    "if not exist models\neu_yolo12n.onnx (",
    "  echo Missing models\neu_yolo12n.onnx",
    "  pause",
    "  exit /b 1",
    ")",
    "bin\industrial_vision_defect.exe --backend onnx --model models\neu_yolo12n.onnx --input-size 640 --input sample_data\images --output results_final_onnx_sample --show 0 --save 1 --evaluate 1",
    "echo.",
    "echo Final ONNX results saved to results_final_onnx_sample",
    "pause"
)

$RunBatchBat = Join-Path $PackageRoot "run_batch_dataset_sample.bat"
Set-Content -LiteralPath $RunBatchBat -Encoding ASCII -Value @(
    "@echo off",
    "cd /d %~dp0",
    "bin\industrial_vision_defect.exe --backend onnx --model models\neu_yolo12n.onnx --input-size 640 --input sample_data\images --output results_batch_dataset_sample --show 0 --save 1 --evaluate 1",
    "echo.",
    "echo Batch detection results saved to results_batch_dataset_sample",
    "pause"
)

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

Compress-Archive -LiteralPath $PackageRoot -DestinationPath $ZipPath -Force
Write-Host "Package directory: $PackageRoot"
Write-Host "Package zip: $ZipPath"
