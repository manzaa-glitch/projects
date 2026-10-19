$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SonyRevision = "e212b0b9974d52d334cc793ef1b8cabaf251e5f8"
$SonyUrl = "git+https://github.com/sony/sampleid.git@$SonyRevision#egg=sampleid"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found at $Python. Create it with: py -3.12 -m venv .venv"
}

Write-Host "Removing the incomplete wheel-style SampleID installation..."
& $Python -m pip uninstall -y sampleid

$EditableClone = Join-Path $RepoRoot ".venv\src\sampleid"
if (Test-Path $EditableClone) {
    Write-Host "Removing stale editable checkout: $EditableClone"
    Remove-Item -Recurse -Force $EditableClone
}

Write-Host "Installing Sony SampleID in editable mode so its checkpoint model modules are retained..."
& $Python -m pip install --no-deps --no-cache-dir -e $SonyUrl

Write-Host "Verifying the model target required by the checkpoint..."
& $Python -c "from sampleid import SampleID; import sampleid; import src.networks.resnet_ibn; print('sampleid:', sampleid.__file__); print('Sony model target import: OK')"

Write-Host ""
Write-Host "Repair complete. Now run:"
Write-Host ".\.venv\Scripts\python.exe diagnose_sampleid.py"
