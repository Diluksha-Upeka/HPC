$ErrorActionPreference = 'Stop'

$compilerPath = 'C:\msys64\ucrt64\bin'
if (Test-Path $compilerPath) {
    $env:PATH = "$compilerPath;$env:PATH"
}

Write-Host '=== Build ===' -ForegroundColor Cyan
g++ -std=c++17 -O2 -Wall -Wextra -o graph_bfs.exe graph_bfs.cpp

Write-Host "`n=== Demo 1: Synthetic Graph ===" -ForegroundColor Cyan
.\graph_bfs.exe 10 0.3 0

Write-Host "`n=== Demo 2: Manual Graph Input ===" -ForegroundColor Cyan
Get-Content .\demo_manual_input.txt | .\graph_bfs.exe --manual

Write-Host "`n=== Demo 3: JSON For Visualizer ===" -ForegroundColor Cyan
Get-Content .\demo_manual_input.txt | .\graph_bfs.exe --manual --json > graph.json
Write-Host 'Generated graph.json'

Write-Host "`n=== Open Visualizer ===" -ForegroundColor Cyan
Start-Process .\visualizer.html
