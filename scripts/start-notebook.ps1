$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'The project environment is missing. Follow SETUP_REPORT.txt first.'
}
$env:JUPYTER_CONFIG_DIR = Join-Path $projectRoot '.jupyter\config'
$env:JUPYTER_DATA_DIR = Join-Path $projectRoot '.jupyter\data'
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot '.jupyter\runtime'
$env:IPYTHONDIR = Join-Path $projectRoot '.jupyter\ipython'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.tools\python'
$env:VIRTUAL_ENV = Join-Path $projectRoot '.venv'
$env:PATH = (Join-Path $projectRoot '.venv\Scripts') + [IO.Path]::PathSeparator + $env:PATH
$env:PYTHONUTF8 = '1'
$env:DEEPEVAL_TELEMETRY_OPT_OUT = 'YES'
$env:DEEPEVAL_UPDATE_WARNING_OPT_OUT = 'YES'
& $projectPython -m jupyterlab --no-browser Full_Code_Notebook.ipynb
exit $LASTEXITCODE
