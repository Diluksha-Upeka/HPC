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

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [OPTIONS] [V] [density] [source]\n"
        << "\n"
        << "Positional arguments (all optional):\n"
        << "  V          Number of vertices          (default: 10)\n"
        << "  density    Edge probability [0.0, 1.0]  (default: 0.3)\n"
        << "  source     BFS starting node            (default: 0)\n"
        << "\n"
        << "Options:\n"
        << "  --manual     Read graph data interactively from stdin\n"
        << "  --json       Output results as JSON (pipe to visualizer)\n"
        << "  --seed N     RNG seed for reproducibility  (default: 42)\n"
        << "  --threads N  Number of OpenMP threads       (default: 4)\n"
        << "  --verify     Run serial BFS and compare for correctness\n"
        << "  --help       Show this help message\n"
        << "\n"
        << "Examples:\n"
        << "  " << program << " 1000 0.01 0\n"
        << "  " << program << " 1000 0.01 0 --threads 8\n"
        << "  " << program << " 1000 0.01 0 --threads 4 --verify\n"
        << "  " << program << " --manual --threads 2\n";
}

void print_text(const Graph& graph, const std::vector<int>& dist, int source,
                double gen_ms, double bfs_ms, int num_threads,
                double serial_ms) {
    const int V = graph.num_vertices;

    if (V <= 200) {
        std::cout << "\nAdjacency list:\n";
        for (int u = 0; u < V; ++u) {
            std::cout << "  " << u << " ->";
            for (std::size_t i = 0; i < graph.adj[u].size(); ++i)
                std::cout << (i == 0 ? " " : ", ") << graph.adj[u][i];
            std::cout << "\n";
        }
    } else {
        std::cout << "\n(Adjacency list omitted for V > 200)\n";
    }

    std::cout << "\nDistance array (hops from node " << source << "):\n";
    if (V <= 200) {
        for (int v = 0; v < V; ++v) {
            std::cout << "  node " << v << " : ";
            if (dist[v] == -1) std::cout << "unreachable";
            else               std::cout << dist[v];
            std::cout << "\n";
        }
    } else {
        int max_dist = 0;
        for (int v = 0; v < V; ++v)
            if (dist[v] > max_dist) max_dist = dist[v];
        std::vector<int> hist(max_dist + 2, 0);
        for (int v = 0; v < V; ++v) {
            if (dist[v] == -1) ++hist[max_dist + 1];
            else               ++hist[dist[v]];
        }
        for (int d = 0; d <= max_dist; ++d)
            if (hist[d] > 0)
                std::cout << "  distance " << d << " : " << hist[d] << " nodes\n";
        if (hist[max_dist + 1] > 0)
            std::cout << "  unreachable : " << hist[max_dist + 1] << " nodes\n";
    }

    int reachable = 0, total_edges = 0;
    for (int v = 0; v < V; ++v) {
        if (dist[v] != -1) ++reachable;
        total_edges += static_cast<int>(graph.adj[v].size());
    }
    total_edges /= 2;

    std::cout << "\nGraph stats: " << total_edges << " edges, "
              << reachable << "/" << V
              << " vertices reachable from node " << source << "\n";
    std::cout << "OpenMP threads: " << num_threads << "\n";
    std::cout << "Timing: graph generation " << gen_ms
              << " ms, parallel BFS " << bfs_ms << " ms\n";

    if (serial_ms > 0.0) {
        double speedup    = serial_ms / bfs_ms;
        double efficiency = speedup / num_threads;
        std::cout << "\n--- Performance Comparison ---\n";
        std::cout << "  Serial BFS time:   " << serial_ms << " ms\n";
        std::cout << "  Parallel BFS time: " << bfs_ms << " ms\n";
        std::cout << "  Speedup (S):       " << speedup << "x\n";
        std::cout << "  Efficiency (E):    " << efficiency
                  << " (" << (efficiency * 100.0) << "%)\n";
    }
}

void print_json(const Graph& graph, const std::vector<int>& dist, int source,
                double gen_ms, double bfs_ms) {
    const int V = graph.num_vertices;
    int total_edges = 0;
    for (int v = 0; v < V; ++v)
        total_edges += static_cast<int>(graph.adj[v].size());
    total_edges /= 2;

    std::cout << "{\n";
    std::cout << "  \"vertices\": " << V << ",\n";
    std::cout << "  \"edges\": " << total_edges << ",\n";
    std::cout << "  \"source\": " << source << ",\n";
    std::cout << "  \"gen_ms\": " << gen_ms << ",\n";
    std::cout << "  \"bfs_ms\": " << bfs_ms << ",\n";

    std::cout << "  \"edge_list\": [";
    bool first = true;
    for (int u = 0; u < V; ++u) {
        for (int v : graph.adj[u]) {
            if (u < v) {
                if (!first) std::cout << ",";
                std::cout << "[" << u << "," << v << "]";
                first = false;
            }
        }
    }
    std::cout << "],\n";

    std::cout << "  \"distance\": [";
    for (int v = 0; v < V; ++v) {
        if (v > 0) std::cout << ",";
        std::cout << dist[v];
    }
    std::cout << "]\n";
    std::cout << "}\n";
}

int main(int argc, char* argv[]) {
    int      V           = 10;
    double   density     = 0.3;
    int      source      = 0;
    uint64_t seed        = 42;
    bool     json        = false;
    bool     manual      = false;
    int      num_threads = 4;
    bool     verify      = false;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--manual") == 0) {
            manual = true;
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
            if (num_threads < 1) num_threads = 1;
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else {
            positional.push_back(argv[i]);
        }
    }

    if (manual) {
        if (positional.size() == 0) {
            if (!prompt_for_manual_config(V, source))
                return 1;
        } else {
            if (positional.size() >= 1) V      = std::stoi(positional[0]);
            if (positional.size() >= 2) source = std::stoi(positional[1]);
        }
    } else {
        if (positional.size() >= 1) V       = std::stoi(positional[0]);
        if (positional.size() >= 2) density = std::stod(positional[1]);
        if (positional.size() >= 3) source  = std::stoi(positional[2]);
    }

    if (V <= 0) { std::cerr << "Error: V must be > 0\n"; return 1; }
    if (source < 0 || source >= V) {
        std::cerr << "Error: source must be in [0, " << V - 1 << "]\n";
        return 1;
    }
    if (!manual && (density < 0.0 || density > 1.0)) {
        std::cerr << "Error: density must be in [0.0, 1.0]\n";
        return 1;
    }

    Graph graph(V);

    auto t0 = std::chrono::high_resolution_clock::now();
    if (manual) {
        if (!json) {
            std::cout << "Manual graph input mode: V=" << V
                      << ", source=" << source << "\n";
        }
        if (!read_manual_graph(graph))
            return 1;
    } else {
        if (!json) {
            std::cout << "Generating undirected graph: V=" << V
                      << ", density=" << density
                      << ", source=" << source
                      << ", seed=" << seed << "\n";
        }
        graph = generate_graph(V, density, seed);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double serial_ms = 0.0;
    std::vector<int> serial_dist;
    if (verify) {
        if (!json) std::cout << "Running serial BFS for verification ...\n";
        auto ts0 = std::chrono::high_resolution_clock::now();
        serial_dist = bfs_serial(graph, source);
        auto ts1 = std::chrono::high_resolution_clock::now();
        serial_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    }

    if (!json) {
        std::cout << "Running parallel BFS (OpenMP, " << num_threads
                  << " threads) from source node " << source << " ...\n";
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<int> dist = bfs_parallel(graph, source, num_threads);
    auto t3 = std::chrono::high_resolution_clock::now();
    double bfs_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (verify) {
        bool correct = (dist == serial_dist);
        if (!json) {
            std::cout << "\nCorrectness check: "
                      << (correct ? "PASSED (parallel == serial)"
                                  : "FAILED (mismatch detected!)")
                      << "\n";
        }
        if (!correct) {
            int shown = 0;
            for (int v = 0; v < V && shown < 10; ++v) {
                if (dist[v] != serial_dist[v]) {
                    std::cerr << "  node " << v
                              << ": parallel=" << dist[v]
                              << " serial=" << serial_dist[v] << "\n";
                    ++shown;
                }
            }
            return 1;
        }
    }

    if (json)
        print_json(graph, dist, source, gen_ms, bfs_ms);
    else
        print_text(graph, dist, source, gen_ms, bfs_ms, num_threads, serial_ms);

    return 0;
}
