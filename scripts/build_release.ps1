param(
    [string]$OpenCvDir = $env:OpenCV_DIR,
    [string]$VsDevCmd = $env:VSDEVCMD
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (-not $VsDevCmd) {
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $VsWhere) {
        $InstallPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($InstallPath) {
            $VsDevCmd = Join-Path $InstallPath "Common7\Tools\VsDevCmd.bat"
        }
    }
}

if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "Cannot find Visual Studio developer command script. Set VSDEVCMD or pass -VsDevCmd."
}

if (-not $OpenCvDir) {
    throw "OpenCV_DIR is not set. Set the OpenCV_DIR environment variable or pass -OpenCvDir."
}

Push-Location $ProjectRoot
try {
    cmd /c "`"$VsDevCmd`" -arch=x64 && cmake -S . -B build -DOpenCV_DIR=$OpenCvDir && cmake --build build --config Release"
} finally {
    Pop-Location
}
