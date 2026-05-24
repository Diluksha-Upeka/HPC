#include <cuda_runtime.h>  // for cuda Malloc, cuda memcpy
#include <iostream>     // print outputs 
#include <vector>       // dynamic arrays
#include <queue>          // this is used in serial bfs
#include <random>         // for random graph generation
#include <chrono>         // To measure  CPU serial BFS time

using namespace std;   // this will allow us to avoid writing std:: before every standard library type or function

#define INF -1

struct Graph {             ////////////////////////////
    int V;
    vector<vector<int>> adj;

    Graph(int vertices) : V(vertices), adj(vertices) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);         ////////////
        adj[v].push_back(u);         ///////////
    }
};

Graph generate_graph(int V, double density, int seed) {         //////////////////// // If you use the same seed, you get the same random graph every time
    Graph g(V);                                        // crate the graph 
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int u = 0; u < V; u++) {
        for (int v = u + 1; v < V; v++) {            // check every possible edge (u, v) 
            if (dist(rng) < density) {                //means how many edges/connections the graph has
                g.add_edge(u, v);
            }
        }
    }

    return g;
}

vector<int> bfs_serial(const Graph& g, int source) {    //////////// Serial BFS on CPU
    vector<int> distance(g.V, INF);        //////////////// distance array initialized to -1 (unvisited)
    queue<int> q;

    distance[source] = 0;     ///// distance to source is 0
    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : g.adj[u]) {
            if (distance[v] == INF) {
                distance[v] = distance[u] + 1;   // update distance for unvisited neighbor
                q.push(v);
            }
        }
    }

    return distance;
}
//compressed spare row
void convert_to_csr(    ////////////////////////////////////////    // flat arrays,Convert graph to CSR format for GPU processing.beacause cuda works better with 
    const Graph& g,
    vector<int>& row_offsets,   /////////// neighbors start and  end 
    vector<int>& col_indices   //////////// all neighbors in a single array
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

__global__ void bfs_cuda_kernel(   //////////////////////// // expand the current BFS frontier in parallel on the GPU.
    const int* row_offsets,         // This global funtion run on the GPU but called from cpu
    const int* col_indices,
    const int* current_frontier,   // current level's nodes 
    int frontier_size,
    int* next_frontier,
    int* next_frontier_size,
    int* distance,
    int level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;      /////////////////// give each thread a unique ID based

    if (tid < frontier_size) {
        int u = current_frontier[tid];

        int start = row_offsets[u];                ///////////// check neighbors 
        int end = row_offsets[u + 1];

        for (int edge = start; edge < end; edge++) {
            int v = col_indices[edge];

            if (atomicCAS(&distance[v], INF, level) == INF) {  //////// two threads may find same vertex.stop race  condition(avoid duplicate visits)
                int pos = atomicAdd(next_frontier_size, 1);    ///////////////////
                next_frontier[pos] = v;
            }
        }
    }
}

////////////////////// this function manage the all BFS process on the GPU
vector<int> bfs_cuda(const Graph& g, int source, double& cuda_time_ms) {  ////////////////////// this function manage the all BFS process on the GPU, including memory allocation, kernel launches, and result retrieval. It returns the distance array after BFS is complete.
    int V = g.V;

    vector<int> row_offsets;
    vector<int> col_indices;
    convert_to_csr(g, row_offsets, col_indices);

    int E = col_indices.size();  // E is the total number of edges. graph is undirected, each edge stored twice 

    vector<int> distance(V, INF);
    distance[source] = 0;

    int* d_row_offsets = nullptr;      // here d means device (GPU) memory pointers
    int* d_col_indices = nullptr;
    int* d_distance = nullptr;
    int* d_current_frontier = nullptr;
    int* d_next_frontier = nullptr;
    int* d_next_frontier_size = nullptr;

    cudaMalloc(&d_row_offsets, (V + 1) * sizeof(int));   ////////////////////// allocate memory on GPU
    cudaMalloc(&d_col_indices, E * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));
    cudaMalloc(&d_current_frontier, V * sizeof(int));
    cudaMalloc(&d_next_frontier, V * sizeof(int));
    cudaMalloc(&d_next_frontier_size, sizeof(int));

    cudaMemcpy(d_row_offsets, row_offsets.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);  ////////////////////// copy data from CPU to GPU
    cudaMemcpy(d_col_indices, col_indices.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    vector<int> current_frontier;  // initially the frontier contains only the source vertex
    current_frontier.push_back(source);

    int level = 1;   // BFS level starts from 1 since source is at level 0

    cudaEvent_t start, stop;  // for measuring GPU execution time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    while (!current_frontier.empty()) {  /////////////// continues until no new vertices are found.
        int frontier_size = current_frontier.size();  // Get frontier size

        cudaMemcpy(                  //Copy current frontier to GPU
            d_current_frontier,
            current_frontier.data(),
            frontier_size * sizeof(int),
            cudaMemcpyHostToDevice
        );

        int zero = 0;              //Reset next frontier size to 0 
        cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

        int threads_per_block = 256;  //Decide GPU blocks and threads
        int blocks = (frontier_size + threads_per_block - 1) / threads_per_block;

        bfs_cuda_kernel<<<blocks, threads_per_block>>>(    //Launch CUDA kernel
            d_row_offsets,
            d_col_indices,
            d_current_frontier,
            frontier_size,
            d_next_frontier,
            d_next_frontier_size,
            d_distance,
            level
        );

        cudaDeviceSynchronize();    // Wait unti GPU to finish the kernel

        int next_size = 0;    //how many vertices were discovered in the next level.
        cudaMemcpy(&next_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

        current_frontier.resize(next_size);   //Copy next frontier back

        if (next_size > 0) {
            cudaMemcpy(
                current_frontier.data(),
                d_next_frontier,
                next_size * sizeof(int),
                cudaMemcpyDeviceToHost
            );
        }

        level++;   // Increment BFS level 
    }

    cudaEventRecord(stop);    //stop timing will stop for GPU
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cuda_time_ms = elapsed_ms;

    cudaMemcpy(distance.data(), d_distance, V * sizeof(int), cudaMemcpyDeviceToHost); //copy final result to back to CPU

    cudaFree(d_row_offsets);   //////////////////////free GPU memory.GPU memory is limited 
    cudaFree(d_col_indices);
    cudaFree(d_distance);
    cudaFree(d_current_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_frontier_size);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return distance;
}

//  Correctness verification
bool verify_result(const vector<int>& a, const vector<int>& b) {//If both arrays are exactly same Correctness: PASSED
    return a == b;
}

int main(int argc, char* argv[]) {  ///////////////// 
    if (argc < 4) {
        cout << "Usage: ./graph_bfs_cuda <vertices> <density> <source> [seed]\n";
        return 1;
    }

    //This converts command-line values from text to numbers.
    int V = stoi(argv[1]);      
    double density = stod(argv[2]);
    int source = stoi(argv[3]);
    int seed = 42;

    if (argc >= 5) {
        seed = stoi(argv[4]);
    }

    cout << "Generating graph..." << endl;

    Graph g = generate_graph(V, density, seed);  // will generaet a random graph

    auto s1 = chrono::high_resolution_clock::now();  // meausre CPU BFS time.
    vector<int> serial_dist = bfs_serial(g, source);
    auto s2 = chrono::high_resolution_clock::now();

    double serial_ms = chrono::duration<double, milli>(s2 - s1).count();

    double cuda_ms = 0.0;     //runs GPU BFS and returns
    vector<int> cuda_dist = bfs_cuda(g, source, cuda_ms);

    cout << "CUDA BFS completed" << endl;
    cout << "Vertices: " << V << endl;
    cout << "Density: " << density << endl;
    cout << "Source: " << source << endl;
    cout << "Serial time: " << serial_ms << " ms" << endl;
    cout << "CUDA time: " << cuda_ms << " ms" << endl;
    cout << "Speedup: " << serial_ms / cuda_ms << "x" << endl;
    cout << "Correctness: " << (verify_result(serial_dist, cuda_dist) ? "PASSED" : "FAILED") << endl; ///////////////////////////////

    return 0;
}
