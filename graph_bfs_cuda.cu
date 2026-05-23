#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>

using namespace std;

#define INF -1

struct Graph {
    int V;
    vector<vector<int>> adj;

    Graph(int vertices) : V(vertices), adj(vertices) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
};

Graph generate_graph(int V, double density, int seed) {
    Graph g(V);
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int u = 0; u < V; u++) {
        for (int v = u + 1; v < V; v++) {
            if (dist(rng) < density) {
                g.add_edge(u, v);
            }
        }
    }

    return g;
}

vector<int> bfs_serial(const Graph& g, int source) {
    vector<int> distance(g.V, INF);
    queue<int> q;

    distance[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : g.adj[u]) {
            if (distance[v] == INF) {
                distance[v] = distance[u] + 1;
                q.push(v);
            }
        }
    }

    return distance;
}

void convert_to_csr(
    const Graph& g,
    vector<int>& row_offsets,
    vector<int>& col_indices
) {
    row_offsets.resize(g.V + 1);
    row_offsets[0] = 0;

    for (int i = 0; i < g.V; i++) {
        row_offsets[i + 1] = row_offsets[i] + g.adj[i].size();

        for (int neighbor : g.adj[i]) {
            col_indices.push_back(neighbor);
        }
    }
}

__global__ void bfs_cuda_kernel(
    const int* row_offsets,
    const int* col_indices,
    const int* current_frontier,
    int frontier_size,
    int* next_frontier,
    int* next_frontier_size,
    int* distance,
    int level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < frontier_size) {
        int u = current_frontier[tid];

        int start = row_offsets[u];
        int end = row_offsets[u + 1];

        for (int edge = start; edge < end; edge++) {
            int v = col_indices[edge];

            if (atomicCAS(&distance[v], INF, level) == INF) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}

vector<int> bfs_cuda(const Graph& g, int source, double& cuda_time_ms) {
    int V = g.V;

    vector<int> row_offsets;
    vector<int> col_indices;
    convert_to_csr(g, row_offsets, col_indices);

    int E = col_indices.size();

    vector<int> distance(V, INF);
    distance[source] = 0;

    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    int* d_distance = nullptr;
    int* d_current_frontier = nullptr;
    int* d_next_frontier = nullptr;
    int* d_next_frontier_size = nullptr;

    cudaMalloc(&d_row_offsets, (V + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, E * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));
    cudaMalloc(&d_current_frontier, V * sizeof(int));
    cudaMalloc(&d_next_frontier, V * sizeof(int));
    cudaMalloc(&d_next_frontier_size, sizeof(int));

    cudaMemcpy(d_row_offsets, row_offsets.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    vector<int> current_frontier;
    current_frontier.push_back(source);

    int level = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    while (!current_frontier.empty()) {
        int frontier_size = current_frontier.size();

        cudaMemcpy(
            d_current_frontier,
            current_frontier.data(),
            frontier_size * sizeof(int),
            cudaMemcpyHostToDevice
        );

        int zero = 0;
        cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

        int threads_per_block = 256;
        int blocks = (frontier_size + threads_per_block - 1) / threads_per_block;

        bfs_cuda_kernel<<<blocks, threads_per_block>>>(
            d_row_offsets,
            d_col_indices,
            d_current_frontier,
            frontier_size,
            d_next_frontier,
            d_next_frontier_size,
            d_distance,
            level
        );

        cudaDeviceSynchronize();

        int next_size = 0;
        cudaMemcpy(&next_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

        current_frontier.resize(next_size);

        if (next_size > 0) {
            cudaMemcpy(
                current_frontier.data(),
                d_next_frontier,
                next_size * sizeof(int),
                cudaMemcpyDeviceToHost
            );
        }

        level++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cuda_time_ms = elapsed_ms;

    cudaMemcpy(distance.data(), d_distance, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_distance);
    cudaFree(d_current_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_frontier_size);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return distance;
}

bool verify_result(const vector<int>& a, const vector<int>& b) {
    return a == b;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: ./graph_bfs_cuda <vertices> <density> <source> [seed]\n";
        return 1;
    }

    int V = stoi(argv[1]);
    double density = stod(argv[2]);
    int source = stoi(argv[3]);
    int seed = 42;

    if (argc >= 5) {
        seed = stoi(argv[4]);
    }

    cout << "Generating graph..." << endl;

    Graph g = generate_graph(V, density, seed);

    auto s1 = chrono::high_resolution_clock::now();
    vector<int> serial_dist = bfs_serial(g, source);
    auto s2 = chrono::high_resolution_clock::now();

    double serial_ms = chrono::duration<double, milli>(s2 - s1).count();

    double cuda_ms = 0.0;
    vector<int> cuda_dist = bfs_cuda(g, source, cuda_ms);

    cout << "CUDA BFS completed" << endl;
    cout << "Vertices: " << V << endl;
    cout << "Density: " << density << endl;
    cout << "Source: " << source << endl;
    cout << "Serial time: " << serial_ms << " ms" << endl;
    cout << "CUDA time: " << cuda_ms << " ms" << endl;
    cout << "Speedup: " << serial_ms / cuda_ms << "x" << endl;
    cout << "Correctness: " << (verify_result(serial_dist, cuda_dist) ? "PASSED" : "FAILED") << endl;

    return 0;
}
