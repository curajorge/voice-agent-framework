param (
    [switch]$Cli
)

$ErrorActionPreference = "Stop"

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "Virtual environment not found. Creating..."
    python -m venv venv
}

Write-Host "Ensuring dependencies are installed..."
.\venv\Scripts\pip install -r requirements.txt | Out-Null

if ($Cli) {
    Write-Host "Starting AgentFramework CLI..."
    & .\venv\Scripts\python -m src.main --cli
} else {
    Write-Host "Starting AgentFramework Server..."
    & .\venv\Scripts\python -m src.main
}
