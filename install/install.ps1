<#
.SYNOPSIS
    Installs the IS3D Blender add-on and pulls its model assets.

.DESCRIPTION
    Finds a Blender 5.3+ installation, drops the add-on into its scripts/addons
    folder, then downloads the engine DLL and the weight file (split into parts),
    verifies every SHA-256 and rebuilds idol_weights.bin.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install/install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install/install.ps1 -BlenderRoot "D:\Apps\Blender 5.3"

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install/install.ps1 -AssetBaseUrl "https://hf-mirror.com/your/repo/resolve/main"
#>
[CmdletBinding()]
param(
    [string]$BlenderRoot,
    [string]$AddonsDir,
    [string]$Version = "0.4.0",
    [string]$Repo = "MARK42IRPC/Idol-Splat-3dGen_BlenderScript",
    [string]$AssetBaseUrl,
    [string]$FromZip,
    [switch]$Source,
    [switch]$SkipAssets,
    [switch]$KeepParts,
    [switch]$SkipVerify,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# The weights this script pulls are an inference-only repack of the upstream
# models below; rebuilding them from source is described in docs/BUILD.md and
# docs/ASSETS.md. Nothing is redistributed inside the git repository.
#
#   idol (image -> latent -> gaussian anchors, splat decoder)
#     https://github.com/speridlabs/idol-sdk        https://huggingface.co/speridlabs
#   FLUX.2 klein VAE (image tokens)
#     https://github.com/black-forest-labs/flux2   https://huggingface.co/black-forest-labs
#   DINOv3 ViT-H/16+ (conditioning)
#     https://github.com/facebookresearch/dinov3   https://huggingface.co/facebook
#   BiRefNet (background matte)
#     https://github.com/ZhengPeng7/BiRefNet       https://huggingface.co/ZhengPeng7
$DefaultAssetBase = "https://github.com/$Repo/releases/download/v$Version"

function Resolve-AssetBase {
    if ($AssetBaseUrl) { return $AssetBaseUrl.TrimEnd("/") }
    return $DefaultAssetBase
}

function Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Note($text) { Write-Host "    $text" -ForegroundColor DarkGray }
function Warn($text) { Write-Host "    ! $text" -ForegroundColor Yellow }

function DropFile($path) {
    if (Test-Path -LiteralPath $path) { [System.IO.File]::Delete($path) }
}

function DropDir($path) {
    if (Test-Path -LiteralPath $path) { [System.IO.Directory]::Delete($path, $true) }
}

function Find-Blender {
    $candidates = @()
    if ($BlenderRoot) {
        $candidates += Get-Item -LiteralPath $BlenderRoot -ErrorAction SilentlyContinue
    } else {
        $globs = @(
            "C:\Program Files (x86)\Steam\steamapps\common\Blender",
            "C:\Program Files\Steam\steamapps\common\Blender",
            "D:\SteamLibrary\steamapps\common\Blender",
            "E:\SteamLibrary\steamapps\common\Blender",
            (Join-Path $env:ProgramFiles "Blender Foundation\Blender*"),
            (Join-Path ${env:ProgramFiles(x86)} "Blender Foundation\Blender*"),
            (Join-Path $env:LOCALAPPDATA "Programs\Blender Foundation\Blender*")
        )
        foreach ($glob in $globs) {
            if (-not $glob) { continue }
            $candidates += Get-Item -Path $glob -ErrorAction SilentlyContinue
        }
    }
    $found = @()
    foreach ($dir in $candidates) {
        if (-not $dir) { continue }
        $exe = Join-Path $dir.FullName "blender.exe"
        if (Test-Path -LiteralPath $exe) {
            $version = [version]"0.0"
            if ($dir.Name -match "(\d+\.\d+)") { $version = [version]$Matches[1] }
            $found += [pscustomobject]@{ Exe = $exe; Root = $dir.FullName; Version = $version }
        }
    }
    return ($found | Sort-Object Version -Descending | Select-Object -First 1)
}

function Get-AddonsDir($blender) {
    if ($AddonsDir) {
        New-Item -ItemType Directory -Force -Path $AddonsDir | Out-Null
        return (Resolve-Path -LiteralPath $AddonsDir).Path
    }
    Step "Locating the add-on folder"
    $expr = "import bpy;print('IS3D_SCRIPTS=' + bpy.utils.user_resource('SCRIPTS'))"
    $lines = & $blender.Exe -b --factory-startup --python-expr $expr 2>&1
    $scripts = $null
    foreach ($line in $lines) {
        if ("$line" -match "IS3D_SCRIPTS=(.+)$") { $scripts = $Matches[1].Trim() }
    }
    if (-not $scripts) {
        $version = "5.3"
        $first = (& $blender.Exe --version 2>&1 | Select-Object -First 1)
        if ("$first" -match "(\d+\.\d+)") { $version = $Matches[1] }
        $scripts = Join-Path $env:APPDATA "Blender Foundation\Blender\$version\scripts"
        Warn "Blender did not report its script path, assuming $scripts"
    }
    $addons = Join-Path $scripts "addons"
    New-Item -ItemType Directory -Force -Path $addons | Out-Null
    Note "addons: $addons"
    return $addons
}

function Get-Remote($url, $target) {
    if (Test-Path -LiteralPath $target) {
        if (-not $Force) { return }
        DropFile $target
    }
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L -sS --fail --retry 3 --retry-delay 2 -C - -o $target $url
        if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
    } else {
        Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing
    }
}

function Get-Sums($path) {
    $map = @{}
    foreach ($line in Get-Content -LiteralPath $path) {
        if ($line -match "^\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*$") { $map[$Matches[2]] = $Matches[1].ToLowerInvariant() }
    }
    return $map
}

function Test-Hash($file, $expected) {
    if ($SkipVerify -or -not $expected) { return $true }
    $actual = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        Warn "checksum mismatch for $(Split-Path -Leaf $file)"
        Note "expected $expected"
        Note "actual   $actual"
        return $false
    }
    return $true
}

# --------------------------------------------------------------------- add-on
$blender = Find-Blender
if (-not $blender) { throw "Blender was not found. Pass -BlenderRoot <dir> or -AddonsDir <dir>." }
if ($blender.Version -lt [version]"1.0") {
    # Steam installs live in a folder called "Blender", so ask the binary.
    $first = (& $blender.Exe --version 2>&1 | Select-Object -First 1)
    if ("$first" -match "(\d+\.\d+)") { $blender.Version = [version]$Matches[1] }
}
Step "Blender $($blender.Version) at $($blender.Root)"
$addons = Get-AddonsDir $blender
$target = Join-Path $addons "is3d"

if ($Source) {
    Step "Installing from this checkout"
    $local = Join-Path (Split-Path -Parent $PSScriptRoot) "is3d"
    if (-not (Test-Path -LiteralPath (Join-Path $local "__init__.py"))) { throw "run -Source from a git clone of the repository" }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item -Path (Join-Path $local "*.py") -Destination $target -Force
    Note "copied python modules to $target"
} elseif ($FromZip) {
    Step "Installing from $FromZip"
    DropDir $target
    Expand-Archive -LiteralPath $FromZip -DestinationPath $addons -Force
} else {
    $zipUrl = (Resolve-AssetBase) + "/is3d-addon-$Version.zip"
    $zip = Join-Path $env:TEMP "is3d-addon-$Version.zip"
    Step "Downloading the add-on"
    Note $zipUrl
    Get-Remote $zipUrl $zip
    DropDir $target
    Expand-Archive -LiteralPath $zip -DestinationPath $addons -Force
    DropFile $zip
}
DropDir (Join-Path $target "__pycache__")

# --------------------------------------------------------------------- assets
$native = Join-Path $target "native"
New-Item -ItemType Directory -Force -Path $native | Out-Null
$dll = Join-Path $native "idol_engine.dll"
$weights = Join-Path $native "idol_weights.bin"

if ($SkipAssets) {
    Step "Skipping model assets (-SkipAssets)"
} elseif ((Test-Path -LiteralPath $weights) -and (Test-Path -LiteralPath $dll) -and -not $Force) {
    $size = [math]::Round((Get-Item -LiteralPath $weights).Length / 1GB, 2)
    Step "Model assets already present ($size GB)"
    Note $native
} else {
    $base = Resolve-AssetBase
    Step "Fetching model assets"
    Note ("{0:N0} bytes of weights, repacked from the upstream models listed in NOTICE.md" -f 3537766912)
    Note $base

    $sumsFile = Join-Path $env:TEMP "is3d-SHA256SUMS.txt"
    Get-Remote "$base/SHA256SUMS.txt" $sumsFile
    $sums = Get-Sums $sumsFile

    Get-Remote "$base/idol_engine.dll" $dll
    if (-not (Test-Hash $dll $sums["idol_engine.dll"])) { throw "idol_engine.dll is corrupt, re-run with -Force" }

    $parts = @($sums.Keys | Where-Object { $_ -like "idol_weights.bin.part*" } | Sort-Object)
    if ($parts.Count -eq 0) { $parts = @("idol_weights.bin") }
    Note ("weight file: " + ($parts -join ", "))

    $local = @()
    foreach ($part in $parts) {
        $dest = Join-Path $env:TEMP ("is3d-" + $part)
        Write-Host "    -> $part"
        Get-Remote "$base/$part" $dest
        if (-not (Test-Hash $dest $sums[$part])) { DropFile $dest; throw "$part is corrupt, re-run with -Force" }
        $local += $dest
    }

    Step "Rebuilding idol_weights.bin"
    $stream = [System.IO.File]::Create($weights)
    try {
        foreach ($part in $local) {
            $input = [System.IO.File]::OpenRead($part)
            try { $input.CopyTo($stream, 4194304) } finally { $input.Dispose() }
        }
    } finally { $stream.Dispose() }
    if (-not (Test-Hash $weights $sums["idol_weights.bin"])) { throw "the rebuilt weight file is corrupt" }
    $size = [math]::Round((Get-Item -LiteralPath $weights).Length / 1GB, 2)
    Note "$size GB written to $weights"

    if (-not $KeepParts) {
        foreach ($part in $local) { DropFile $part }
        DropFile $sumsFile
    }
}

Write-Host ""
Write-Host "IS3D installed into $target" -ForegroundColor Green
Write-Host "Next: Blender > Edit > Preferences > Add-ons > search 'IS3D' > enable it." -ForegroundColor Green
Write-Host "The panel lives in the 3D viewport sidebar (N) under the IS3D tab."
