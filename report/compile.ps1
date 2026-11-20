# Windows equivalent of compile.sh; run from any working directory.
$ErrorActionPreference = 'Stop'
$reportRoot = $PSScriptRoot
$texBin = Join-Path $reportRoot '..\.local\TinyTeX\bin\windows'
$environmentFile = Join-Path $reportRoot '..\.local\tex-environment.json'
if (Test-Path -LiteralPath $environmentFile) {
    $installation = Get-Content -LiteralPath $environmentFile -Raw | ConvertFrom-Json
    $texBin = Join-Path $installation.texRoot 'bin\windows'
}
if (Test-Path -LiteralPath (Join-Path $texBin 'pdflatex.exe')) {
    # kpathsea needs ASCII paths on Windows installations with Vietnamese names.
    $fileSystem = New-Object -ComObject Scripting.FileSystemObject
    $texRoot = $fileSystem.GetFolder((Resolve-Path "$texBin\..\..").Path).ShortPath.Replace('\', '/')
    $texBin = "$texRoot/bin/windows"
    $reportRoot = $fileSystem.GetFolder($reportRoot).ShortPath
    $env:TEXMFCNF = "$texRoot;$texRoot/texmf-dist/web2c"
    $env:TEXMFROOT = $texRoot
    $env:TEXMFVAR = "$texRoot/texmf-var"
    $env:TEXMFCONFIG = "$texRoot/texmf-config"
    $env:PATH = "$texBin;$env:PATH"
}
foreach ($tool in @('pdflatex', 'bibtex')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing $tool. Install TinyTeX in .local/TinyTeX or add TeX Live to PATH."
    }
}
$buildRelative = 'report_compiled/build'
$buildRoot = Join-Path $reportRoot $buildRelative
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
function Invoke-PdfLatex {
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "-jobname=$buildRelative/main" main.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex failed. See $buildRoot/main.log" }
}
Push-Location -LiteralPath $reportRoot
try {
    Invoke-PdfLatex
    if (Test-Path -LiteralPath "$buildRoot/main.bcf") {
        if (-not (Get-Command biber -ErrorAction SilentlyContinue)) { throw 'Missing biber.' }
        & biber "$buildRelative/main"
    } else {
        & bibtex "$buildRelative/main"
    }
    if ($LASTEXITCODE -ne 0) { throw 'Bibliography compilation failed.' }
    Invoke-PdfLatex
    Invoke-PdfLatex
    Copy-Item -LiteralPath "$buildRoot/main.pdf" -Destination (Join-Path $reportRoot 'Project03_Report.pdf') -Force
    Write-Output "PDF: $(Join-Path $reportRoot 'Project03_Report.pdf')"
} finally {
    Pop-Location
}
