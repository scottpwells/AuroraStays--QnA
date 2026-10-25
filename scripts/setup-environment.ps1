$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.tools\python'
$env:PYTHONUTF8 = '1'
$env:JUPYTER_CONFIG_DIR = Join-Path $projectRoot '.jupyter\config'
$env:JUPYTER_DATA_DIR = Join-Path $projectRoot '.jupyter\data'
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot '.jupyter\runtime'
$env:IPYTHONDIR = Join-Path $projectRoot '.jupyter\ipython'
$projectUv = Join-Path $projectRoot '.tools\bin\uv.exe'
if (-not (Test-Path -LiteralPath $projectUv)) {
    & python -m pip install --target .tools uv==0.13.5 --disable-pip-version-check --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw 'Could not download the project uv tool.' }
}
& $projectUv python install 3.12.10 --install-dir .tools/python --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw 'Could not install project Python.' }
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    & $projectUv venv .venv --python 3.12.10 --no-python-downloads
    if ($LASTEXITCODE -ne 0) { throw 'Could not create project environment.' }
}
& $projectPython -c 'import sys; assert sys.version_info[:3] == (3, 12, 10), "Expected Python 3.12.10"'
if ($LASTEXITCODE -ne 0) { throw 'The existing .venv uses a different Python version.' }
& $projectUv pip sync requirements-notebook.lock.txt --python $projectPython --torch-backend cpu --only-binary :all:
if ($LASTEXITCODE -ne 0) { throw 'Could not install locked dependencies.' }
& $projectPython -m ipykernel install --prefix .venv --name aurorastays-312 --display-name 'AuroraStays (Python 3.12.10)'
if ($LASTEXITCODE -ne 0) { throw 'Could not register project kernel.' }
& $projectUv pip check --python $projectPython
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
Write-Output 'Environment prepared. Run .\scripts\verify-environment.py using .venv\Scripts\python.exe, then .\scripts\start-notebook.ps1.'
