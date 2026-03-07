# HPC Graph BFS Baseline

This project provides a foundational C++ graph-processing baseline for future HPC work. It includes:

- an undirected graph stored as an adjacency list,
- synthetic graph generation with configurable vertex count and edge density,
- a serial Breadth-First Search (BFS) implementation,
- manual graph input mode for correctness demos,
- JSON output for a lightweight HTML visualizer.

## Repository Contents

- `graph_bfs.cpp` - main C++ program
- `visualizer.html` - browser-based graph/BFS visualizer
- `demo.ps1` - standard demo script for evaluation/presentation
- `demo_manual_input.txt` - sample manual graph input used by the demo

## Requirements

On Windows, this project expects a C++17 compiler. The current setup uses MSYS2 MinGW:

- `C:\msys64\ucrt64\bin\g++.exe`

If that compiler directory is not already in `PATH`, `demo.ps1` adds it for the current PowerShell session.

## Build

From `F:\PROJECTS\HPC`:

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -o graph_bfs.exe graph_bfs.cpp
```

## Standard Way To Demonstrate The Project

```powershell
.\demo.ps1
```

That script performs the standard presentation flow:

1. builds the program,
2. runs a synthetic graph example,
3. runs a manual graph example,
4. generates `graph.json`,
5. opens the browser visualizer.


## Running Individual Modes

### 1. Synthetic Graph Mode

```powershell
.\graph_bfs.exe 10 0.3 0
```

Arguments:

- first: number of vertices
- second: edge density in `[0.0, 1.0]`
- third: BFS source vertex

Example:

```powershell
.\graph_bfs.exe 100 0.02 5
```

### 2. Manual Graph Mode

Simplest form:

```powershell
.\graph_bfs.exe --manual
```

The program prompts for:

- number of vertices,
- BFS source vertex,
- number of undirected edges,
- each edge as `u v`.

Example interactive session:

```text
Enter number of vertices: 5
Enter BFS source vertex [0, 4]: 0
Enter number of undirected edges: 5
Enter edges as pairs: u v
Example: 0 3
0 1
0 2
1 3
2 3
3 4
```

### 3. Manual Graph From File

The sample file `demo_manual_input.txt` contains a full manual-input session:

```text
5
0
5
0 1
0 2
1 3
2 3
3 4
```

Meaning:

- line 1: number of vertices,
- line 2: BFS source vertex,
- line 3: number of undirected edges,
- remaining lines: the edge list.

Run it with:

```powershell
Get-Content .\demo_manual_input.txt | .\graph_bfs.exe --manual
```

## JSON Output And Visualizer

To generate JSON output for visualization:

```powershell
Get-Content .\demo_manual_input.txt | .\graph_bfs.exe --manual --json > graph.json
```

Then open:

- `visualizer.html`

In the visualizer you can:

- load `graph.json`,
- inspect graph statistics,
- view BFS distance coloring,
- hover nodes for details,
- animate BFS progression.

## Output Summary

The program reports:

- adjacency list,
- BFS distance array,
- unreachable vertices as `-1` or `unreachable`,
- total edge count,
- number of reachable vertices,
- generation and BFS timings.

## Current Scope

This is the serial baseline implementation. Its purpose is:

- correctness reference,
- baseline timing reference,
- input/output foundation for future OpenMP and MPI versions.
