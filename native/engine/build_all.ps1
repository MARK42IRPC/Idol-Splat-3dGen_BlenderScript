param([switch]$SkipPtx)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $SkipPtx) {
    & "$root\.reference\venv\Scripts\python.exe" -B "$PSScriptRoot\build_ptx.py"
    if ($LASTEXITCODE -ne 0) { throw "ptx" }
    & "$root\.reference\venv\Scripts\python.exe" -B "$PSScriptRoot\embed_ptx.py" "$PSScriptRoot\build\kernels.ptx" "$PSScriptRoot\build\ptx_embed.inc"
    if ($LASTEXITCODE -ne 0) { throw "embed" }
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$inst = & $vswhere -latest -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vc = Join-Path $inst "VC\Auxiliary\Build\vcvars64.bat"
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$cmd = 'call "{0}" >nul && cl /nologo /O2 /MT /EHsc /std:c++17 /utf-8 /LD /DWIN32_LEAN_AND_MEAN /DIDOL_ENGINE_BUILD /I"{1}\include" /Ibuild engine.cpp /Fe:build\idol_engine.dll /Fobuild\engine.obj /link /IMPLIB:build\idol_engine.lib "{2}"' -f $vc,$cuda,(Join-Path $cuda "lib\x64\cuda.lib")
Push-Location $PSScriptRoot
try { & cmd.exe /d /c $cmd } finally { Pop-Location }
