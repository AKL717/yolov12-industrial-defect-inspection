$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvDir = Join-Path $ProjectRoot ".venv-yolo"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    python -m venv $VenvDir
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements-yolo.txt")

Write-Host "YOLO environment ready: $PythonExe"
Write-Host "Train:  & $PythonExe scripts/train_yolo12_neu.py"
Write-Host "Export: & $PythonExe scripts/export_yolo12_onnx.py"
