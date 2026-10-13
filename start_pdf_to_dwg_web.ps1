$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv-ml\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw '프로젝트 .venv-ml 환경이 없습니다. requirements.txt를 설치해 주세요.'
}
& $python -m streamlit run pdf_to_dwg_web.py --server.address 127.0.0.1 --server.port 8502
