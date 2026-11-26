$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$pythonCommand = if (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} else {
    throw "Python 3.10 이상 설치가 필요합니다."
}

Write-Host "[1/3] Checking public bundle..."
& $pythonCommand preflight_public.py
if ($LASTEXITCODE -ne 0) { throw "Privacy/data verification failed." }
Write-Host "[2/3] Installing Streamlit dependencies..."
& $pythonCommand -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
Write-Host "[3/3] Starting app at http://localhost:8501"
& $pythonCommand -m streamlit run app.py
