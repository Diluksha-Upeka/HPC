# HPC Graph BFS — Serial, OpenMP & MPI

Parallel implementation of Breadth-First Search (BFS) using shared and distributed memory models for EC7207 High Performance Computing.

This project includes:

- an undirected graph stored as an adjacency list,
- synthetic graph generation with configurable vertex count and edge density,
- a **serial** BFS baseline implementation,
- a **shared-memory parallel** BFS using OpenMP,
- a **distributed-memory parallel** BFS using MPI,
- manual graph input mode for correctness demos,
- JSON output for a lightweight HTML visualizer.

## Repository Contents

- `graph_bfs.cpp` - serial BFS baseline
- `graph_bfs_omp.cpp` - OpenMP parallel BFS (shared memory)
- `graph_bfs_mpi.cpp` - MPI parallel BFS (distributed memory)
- `visualizer.html` - browser-based graph/BFS visualizer
- `demo.ps1` - standard demo script for evaluation/presentation
- `demo_manual_input.txt` - sample manual graph input used by the demo

## Requirements

On Windows, this project expects a C++17 compiler. The current setup uses MSYS2 MinGW:

- `C:\msys64\ucrt64\bin\g++.exe`
- MPI: MS-MPI or MSYS2 `mingw-w64-ucrt-x86_64-msmpi`

On Linux / HPC cluster:

- `g++` (GCC 7+), `mpicxx`, and optionally OpenMP support

## Build

### Serial

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -o graph_bfs.exe graph_bfs.cpp
```

### OpenMP

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -fopenmp -o graph_bfs_omp.exe graph_bfs_omp.cpp
```

### MPI

```bash
# Linux / HPC cluster
mpicxx -std=c++17 -O2 -Wall -Wextra -o graph_bfs_mpi graph_bfs_mpi.cpp
```

```powershell
# Windows (MS-MPI)
g++ -std=c++17 -O2 -Wall -Wextra -I"$env:MSMPI_INC" -o graph_bfs_mpi.exe graph_bfs_mpi.cpp -L"$env:MSMPI_LIB64" -lmsmpi
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

## OpenMP Program

Build:

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -fopenmp -o graph_bfs_omp.exe graph_bfs_omp.cpp
```

Run a synthetic graph:

```powershell
.\graph_bfs_omp.exe 1000 0.01 0 --threads 4
```

Run with serial verification:

```powershell
.\graph_bfs_omp.exe 1000 0.01 0 --threads 4 --verify
```

Run manual mode:

```powershell
Get-Content .\demo_manual_input.txt | .\graph_bfs_omp.exe --manual --threads 4
```

Generate JSON from the OpenMP program:

```powershell
Get-Content .\demo_manual_input.txt | .\graph_bfs_omp.exe --manual --json > graph.json
```

Supported OpenMP options:

- `--threads N` to choose the number of OpenMP threads,
- `--verify` to compare parallel BFS against the serial BFS result,
- `--manual` to read graph input from stdin,
- `--json` to emit JSON for the visualizer,
- `--seed N` to set the random generator seed.

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

## MPI BFS

### Running

```bash
# 4 MPI processes, 1000 vertices, 1% density, BFS from node 0
mpiexec -n 4 ./graph_bfs_mpi 1000 0.01 0

# With correctness verification against serial BFS
mpiexec -n 4 ./graph_bfs_mpi 1000 0.01 0 --verify

# JSON output for visualizer
mpiexec -n 1 ./graph_bfs_mpi 30 0.15 0 --json > graph.json
```

### How It Works

The MPI version uses a **level-synchronous** approach with **block vertex partitioning**:

1. All ranks generate the same graph (identical seed).
2. Vertices are block-partitioned: rank `r` owns `[r*V/P, (r+1)*V/P)`.
3. At each BFS level, all ranks expand the shared frontier but only claim
   unvisited neighbours within their owned range — no conflicts.
4. `MPI_Allgatherv` merges per-rank discoveries into the next frontier.
5. All ranks update their distance arrays from the merged frontier.

### Timing Breakdown

The MPI version reports:

- **Computation time** — frontier expansion + distance updates
- **Communication time** — `MPI_Allgather` + `MPI_Allgatherv` calls
- **Communication overhead** — percentage of BFS time spent in MPI calls
- **Speedup & efficiency** — when `--verify` is used (compares to serial)

## Current Scope

Implemented:

- Serial BFS baseline (`graph_bfs.cpp`)
- OpenMP shared-memory parallel BFS (`graph_bfs_omp.cpp`)
- MPI distributed-memory parallel BFS (`graph_bfs_mpi.cpp`)

Planned:

- Hybrid OpenMP + CUDA implementation
- Automated benchmarking and scalability analysis
