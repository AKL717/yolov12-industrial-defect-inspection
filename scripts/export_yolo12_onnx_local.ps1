param(
    [string]$PythonExe = $env:PYTHON,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$UltralyticsSrc = Join-Path $ProjectRoot "third_party\ultralytics"

if (-not $PythonExe) {
    $PythonExe = "python"
}

if (-not (Test-Path -LiteralPath $UltralyticsSrc)) {
    throw "Ultralytics source not found. Run: git clone https://github.com/ultralytics/ultralytics.git third_party/ultralytics"
}

$env:PYTHONPATH = "$UltralyticsSrc;$env:PYTHONPATH"

Push-Location $ProjectRoot
try {
    & $PythonExe scripts/export_yolo12_onnx.py @RemainingArgs
} finally {
    Pop-Location
}
