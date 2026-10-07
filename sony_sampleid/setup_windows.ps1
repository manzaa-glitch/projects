$ErrorActionPreference = "Stop"

Write-Host "Setting up Sony SampleID in editable mode..."

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip

if (-not (Test-Path "vendor\sampleid")) {
    New-Item -ItemType Directory -Force -Path "vendor" | Out-Null
    git clone https://github.com/sony/sampleid.git vendor\sampleid
}

& .\.venv\Scripts\python.exe -m pip uninstall -y sampleid
& .\.venv\Scripts\python.exe -m pip install -e .\vendor\sampleid
& .\.venv\Scripts\python.exe -m pip install librosa pandas streamlit

Write-Host ""
Write-Host "Setup complete."
Write-Host "Launch with:"
Write-Host ".\.venv\Scripts\python.exe -m streamlit run app.py"
