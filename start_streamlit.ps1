$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv-ml\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Project Python was not found. Create .venv-ml or .venv and install requirements.txt.'
}
$taskTemp = Join-Path $PSScriptRoot '.runtime-tmp'
New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
Write-Host "Using Python: $taskPython"
& $taskPython -X utf8 check_runtime.py
if ($LASTEXITCODE -ne 0) {
    throw 'Runtime/dependency check failed. Run: .\.venv-ml\Scripts\python.exe -m pip install -r requirements.txt -r requirements-preview.txt'
}
& $taskPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
