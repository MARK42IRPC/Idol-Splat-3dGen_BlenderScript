param([string]$CudaRoot = $env:CUDA_PATH_V12_6, [string]$Python = "python", [string]$VcVars)
$ErrorActionPreference = "Stop"
if (!$CudaRoot) { $CudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$CudaRoot = (Resolve-Path -LiteralPath $CudaRoot).Path
if (!$VcVars) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $installation = & $vswhere -latest -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$installation) { throw "Visual Studio 2022 C++ build tools were not found." }
    $VcVars = Join-Path $installation "VC\Auxiliary\Build\vcvars64.bat"
}
$VcVars = (Resolve-Path -LiteralPath $VcVars).Path
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path "build" -Force | Out-Null
    & $Python -B build_ptx.py --cuda $CudaRoot
    if ($LASTEXITCODE -ne 0) { throw "PTX compilation failed." }
    & $Python -B embed_ptx.py build\kernels.ptx build\ptx_embed.inc
    if ($LASTEXITCODE -ne 0) { throw "PTX embedding failed." }
    $compile = 'call "{0}" >nul && cl /nologo /O2 /MT /EHsc /std:c++17 /utf-8 /LD /DWIN32_LEAN_AND_MEAN /DIDOL_ENGINE_BUILD /I"{1}\include" /Ibuild engine.cpp /Fe:build\idol_engine.dll /Fobuild\engine.obj /link /IMPLIB:build\idol_engine.lib "{2}"' -f $VcVars,$CudaRoot,(Join-Path $CudaRoot "lib\x64\cuda.lib")
    & cmd.exe /d /c $compile
    if ($LASTEXITCODE -ne 0) { throw "Engine build failed." }
} finally {
    Pop-Location
}

