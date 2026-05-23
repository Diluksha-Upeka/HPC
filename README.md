# HPC Graph BFS — Serial, OpenMP, MPI, Hybrid & CUDA

Parallel implementation of Breadth-First Search (BFS) using shared memory, distributed memory, and GPU acceleration models for EC7207 High Performance Computing.

This project includes:

- an undirected graph stored as an adjacency list,
- synthetic graph generation with configurable vertex count and edge density,
- a **serial** BFS baseline implementation,
- a **shared-memory parallel** BFS using OpenMP,
- a **distributed-memory parallel** BFS using MPI,
- a **hybrid** BFS using MPI + OpenMP,
- a **GPU-accelerated** BFS using CUDA,
- manual graph input mode for correctness demos,
- JSON output for a comprehensive browser-based HTML visualizer,
- a Google Colab notebook for benchmarking across environments.

## Repository Contents

- `graph_bfs.cpp` - serial BFS baseline
- `graph_bfs_omp.cpp` - OpenMP parallel BFS (shared memory)
- `graph_bfs_mpi.cpp` - MPI parallel BFS (distributed memory)
- `graph_bfs_hybrid.cpp` - Hybrid MPI+OpenMP parallel BFS
- `graph_bfs_cuda.cu` - CUDA GPU-accelerated BFS
- `HPC_Colab_Benchmark.ipynb` - Google Colab notebook to run performance benchmarks
- `algo_visualizer.html`, `.js`, `.css` - modular, browser-based graph and performance visualizer
- `demo.ps1` - standard demo script for evaluation/presentation
- `demo_manual_input.txt` - sample manual graph input used by the demo

## Requirements

On Windows, this project expects a C++17 compiler. The current setup uses MSYS2 MinGW:

- `C:\msys64\ucrt64\bin\g++.exe`
- MPI: MS-MPI or MSYS2 `mingw-w64-ucrt-x86_64-msmpi`

On Linux / HPC cluster:

- `g++` (GCC 7+), `mpicxx`, optionally OpenMP support
- `nvcc` for CUDA compilation

## Build

### Serial

```powershell
g++ -std=c++17 -O3 -Wall -Wextra -o graph_bfs.exe graph_bfs.cpp
```

### OpenMP

```powershell
g++ -std=c++17 -O3 -Wall -Wextra -fopenmp -o graph_bfs_omp.exe graph_bfs_omp.cpp
```

### MPI

```bash
# Linux / HPC cluster
mpicxx -std=c++17 -O3 -Wall -Wextra -o graph_bfs_mpi graph_bfs_mpi.cpp
```

### Hybrid (MPI + OpenMP)

```bash
# Linux / HPC cluster
mpicxx -std=c++17 -O3 -Wall -Wextra -fopenmp -o graph_bfs_hybrid graph_bfs_hybrid.cpp
```

### CUDA

```bash
# Linux / Colab
nvcc -O3 -std=c++11 -o graph_bfs_cuda graph_bfs_cuda.cu
```

## Running Benchmarks (Google Colab)

The easiest way to run and benchmark all 5 implementations is via the included Google Colab notebook (`HPC_Colab_Benchmark.ipynb`).
It automatically installs dependencies, compiles all targets, executes them with a large graph (e.g., 30k vertices, 5% density), and prints a unified performance report. It also generates the `graph_colab.json` file containing all the execution times for the visualizer.

## Visualizer

To explore the graph structure, BFS levels, and compare algorithm performance, use the provided visualizer:

1. Open `algo_visualizer.html` in your web browser.
2. Load a JSON file (e.g., `graph_colab.json` downloaded from the Colab notebook).
3. Switch between tabs to see detailed explanations and interactive animations of each parallel strategy.
4. Go to the **Performance** tab to view comparative charts based on the loaded JSON.

## Running Locally

### 1. Synthetic Graph Mode

```powershell
.\graph_bfs.exe 100 0.02 5
```

Arguments:
- first: number of vertices
- second: edge density in `[0.0, 1.0]`
- third: BFS source vertex

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

### 3. Verification

Most executables support a `--verify` flag to run the serial algorithm locally and compare the results for correctness:

```powershell
.\graph_bfs_omp.exe 1000 0.01 0 --threads 4 --verify
mpiexec -n 4 ./graph_bfs_mpi 1000 0.01 0 --verify
```

## Implementation Highlights

- **OpenMP:** Uses shared-memory `atomicCAS` on the distance array to securely divide frontier chunks among threads without locks.
- **MPI:** Uses a level-synchronous approach with block vertex partitioning and `MPI_Allgatherv` to merge per-rank discoveries.
- **Hybrid:** Reduces MPI communicator overhead by mapping MPI ranks to distinct CPU sockets, and OpenMP threads to local cores, merging work locally before broadcasting.
- **CUDA:** Maps individual CUDA threads to frontier vertices inside the CSR format, leveraging thousands of GPU cores.

## Current Scope

Implemented:
- Serial BFS baseline (`graph_bfs.cpp`)
- OpenMP shared-memory parallel BFS (`graph_bfs_omp.cpp`)
- MPI distributed-memory parallel BFS (`graph_bfs_mpi.cpp`)
- Hybrid OpenMP + MPI (`graph_bfs_hybrid.cpp`)
- CUDA GPU acceleration (`graph_bfs_cuda.cu`)
- Automated benchmarking script (`HPC_Colab_Benchmark.ipynb`)
- Web-based Visualizer (`algo_visualizer.html`)
