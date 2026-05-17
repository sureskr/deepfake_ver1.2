# ML HTTP API on 127.0.0.1:8000 for TeamsCallingBot (Windows).
# Run in elevated or normal PowerShell. First run installs deps (slow).

$ErrorActionPreference = "Stop"

$DeployRoot = $PSScriptRoot
$EngineRoot = Split-Path -Parent $DeployRoot
Set-Location $EngineRoot

$InferencePy = Join-Path $EngineRoot "inference.py"
if ((Test-Path $InferencePy) -and (Select-String -Path $InferencePy -Pattern "from transformers import Wav2Vec2Processor" -Quiet)) {
    Write-Host ""
    Write-Host "DEPLOYMENT ERROR: $InferencePy still uses Wav2Vec2Processor (broken with transformers 5.x on Windows)." -ForegroundColor Red
    Write-Host "Fix: copy windows_deployment\inference.py -> $InferencePy (overwrite). See COPY_TO_SERVER.txt" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$VenvPython = Join-Path $EngineRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $EngineRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating venv at $EngineRoot\.venv ..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    }
    if (-not (Test-Path $VenvPython)) {
        python -m venv .venv
    }
}

# python -m pip avoids pip.exe self-replace warning on Windows
& $VenvPython -m pip install --upgrade pip
Write-Host "Installing ml_engine requirements (torch/transformers may take several minutes) ..."
& $VenvPip install -r (Join-Path $EngineRoot "requirements.txt")
Write-Host "Installing API requirements ..."
& $VenvPip install -r (Join-Path $DeployRoot "requirements-server.txt")

# Optional overrides (uncomment on VM):
# $env:ML_MODEL_PATH = "C:\MlEngine\models\finetuned\best_model.pth"
# $env:ML_THRESHOLD = "0.7"

# Local Wav2Vec snapshot (avoids Wav2Vec2Processor tokenizer issues; uses local_files_only)
$DefaultLocalW2v = Join-Path $EngineRoot "models\wav2vec2-large-960h"
if (Test-Path $DefaultLocalW2v) {
    $env:ML_WAV2VEC_LOCAL = $DefaultLocalW2v
    Write-Host "ML_WAV2VEC_LOCAL=$($env:ML_WAV2VEC_LOCAL)"
}

$env:PYTHONPATH = $EngineRoot

$UvicornArgs = @(
    "-m", "uvicorn",
    "windows_deployment.api_server:app",
    "--host", "127.0.0.1",
    "--port", "8000"
)

Write-Host "Starting API at http://127.0.0.1:8000 (Ctrl+C to stop) ..."
& $VenvPython @UvicornArgs
