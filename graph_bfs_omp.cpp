// Parallel BFS using OpenMP (level-synchronous approach)

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

// Adjacency list representation
struct Graph {
    int num_vertices;
    std::vector<std::vector<int>> adj;

    explicit Graph(int V) : num_vertices(V), adj(V) {}
};

bool add_undirected_edge(Graph& graph, int u, int v) {
    if (u < 0 || v < 0 || u >= graph.num_vertices || v >= graph.num_vertices || u == v)
        return false;
    graph.adj[u].push_back(v);
    graph.adj[v].push_back(u);
    return true;
}

// Generate random undirected graph with given edge probability
Graph generate_graph(int V, double density, uint64_t seed = 42) {
    Graph g(V);
    if (density <= 0.0) return g;
    if (density >= 1.0) density = 1.0;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int u = 0; u < V; ++u) {
        for (int v = u + 1; v < V; ++v) {
            if (dist(rng) < density) {
                g.adj[u].push_back(v);
                g.adj[v].push_back(u);
            }
        }
    }
    return g;
}

// Serial BFS - used as correctness reference
std::vector<int> bfs_serial(const Graph& graph, int source) {
    const int V = graph.num_vertices;
    std::vector<int> distance(V, -1);
    std::queue<int> frontier;

    distance[source] = 0;
    frontier.push(source);

    while (!frontier.empty()) {
        int current = frontier.front();
        frontier.pop();
        int next_dist = distance[current] + 1;
        for (int neighbor : graph.adj[current]) {
            if (distance[neighbor] == -1) {
                distance[neighbor] = next_dist;
                frontier.push(neighbor);
            }
        }
    }
    return distance;
}

// Parallel BFS using OpenMP
// Expands all nodes at each BFS level concurrently, syncs before next level
std::vector<int> bfs_parallel(const Graph& graph, int source, int num_threads) {
    const int V = graph.num_vertices;
    std::vector<int> distance(V, -1);

    omp_set_num_threads(num_threads);

    distance[source] = 0;

    std::vector<int> current_frontier;
    current_frontier.push_back(source);

    int level = 0;

    while (!current_frontier.empty()) {
        level++;
        const int frontier_size = static_cast<int>(current_frontier.size());

        // Each thread collects discovered nodes in its own buffer
        std::vector<std::vector<int>> thread_local_next(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::vector<int>& local_next = thread_local_next[tid];

            #pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < frontier_size; ++i) {
                int current = current_frontier[i];

                for (int neighbor : graph.adj[current]) {
                    // Atomic CAS to prevent race conditions
                    int expected = -1;
                    if (__atomic_compare_exchange_n(
                            &distance[neighbor],
                            &expected,
                            level,
                            false,
                            __ATOMIC_RELAXED,
                            __ATOMIC_RELAXED)) {
                        local_next.push_back(neighbor);
                    }
                }
            }
        }

        // Merge all thread-local buffers into next frontier
        current_frontier.clear();
        for (auto& local : thread_local_next) {
            current_frontier.insert(current_frontier.end(),
                                    local.begin(), local.end());
        }
    }

    return distance;
}

bool prompt_for_manual_config(int& V, int& source) {
    std::cerr << "Enter number of vertices: ";
    if (!(std::cin >> V) || V <= 0) {
        std::cerr << "Error: number of vertices must be > 0.\n";
        return false;
    }
    std::cerr << "Enter BFS source vertex [0, " << V - 1 << "]: ";
    if (!(std::cin >> source) || source < 0 || source >= V) {
        std::cerr << "Error: source must be in [0, " << V - 1 << "].\n";
        return false;
    }
    return true;
}

bool read_manual_graph(Graph& graph) {
    std::cerr << "Enter number of undirected edges: ";
    int edge_count = 0;
    if (!(std::cin >> edge_count) || edge_count < 0) {
        std::cerr << "Error: edge count must be a non-negative integer.\n";
        return false;
    }
    std::cerr << "Enter edges as pairs: u v\n";
    std::cerr << "Example: 0 3\n";

    for (int edge_index = 0; edge_index < edge_count; ++edge_index) {
        int u = -1, v = -1;
        if (!(std::cin >> u >> v)) {
            std::cerr << "Error: failed to read edge " << edge_index
                      << ". Expected two integers.\n";
            return false;
        }
        if (!add_undirected_edge(graph, u, v)) {
            std::cerr << "Error: invalid edge (" << u << ", " << v << ")"
                      << ". Vertices must be distinct and in [0, "
                      << graph.num_vertices - 1 << "].\n";
            return false;
        }
    }
    return true;
}

// Temporary main for testing core algorithm
// Usage: ./graph_bfs_omp [V] [density] [source] [threads]
int main(int argc, char* argv[]) {
    int      V           = 1000;
    double   density     = 0.01;
    int      source      = 0;
    int      num_threads = 4;
    uint64_t seed        = 42;

    if (argc >= 2) V           = std::stoi(argv[1]);
    if (argc >= 3) density     = std::stod(argv[2]);
    if (argc >= 4) source      = std::stoi(argv[3]);
    if (argc >= 5) num_threads = std::stoi(argv[4]);

    std::cout << "Generating graph: V=" << V
              << ", density=" << density
              << ", source=" << source
              << ", seed=" << seed << "\n";

    Graph graph = generate_graph(V, density, seed);

    // Serial BFS
    auto ts0 = std::chrono::high_resolution_clock::now();
    std::vector<int> serial_dist = bfs_serial(graph, source);
    auto ts1 = std::chrono::high_resolution_clock::now();
    double serial_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();

    // Parallel BFS
    auto tp0 = std::chrono::high_resolution_clock::now();
    std::vector<int> par_dist = bfs_parallel(graph, source, num_threads);
    auto tp1 = std::chrono::high_resolution_clock::now();
    double par_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    // Correctness check
    bool correct = (par_dist == serial_dist);
    std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\n";

    std::cout << "Serial BFS:   " << serial_ms << " ms\n";
    std::cout << "Parallel BFS: " << par_ms << " ms (" << num_threads << " threads)\n";
    std::cout << "Speedup:      " << serial_ms / par_ms << "x\n";

    return correct ? 0 : 1;
}
