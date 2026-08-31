# setup_and_run.ps1
$ErrorActionPreference = "Stop"

# Navigate to the POC directory where the app is located
$pocPath = Join-Path -Path $PSScriptRoot -ChildPath "POC"
if (Test-Path -Path $pocPath) {
    Set-Location -Path $pocPath
} else {
    Write-Warning "POC directory not found. Running in current directory..."
}

# Install dependencies from requirements.txt
if (Test-Path -Path "requirements.txt") {
    Write-Host "Installing requirements..." -ForegroundColor Cyan
    pip install -r requirements.txt
} else {
    Write-Warning "requirements.txt not found! Skipping dependency installation."
}

# Run the Streamlit application
Write-Host "Starting the Streamlit Proof of Concept app..." -ForegroundColor Green
streamlit run app.py
