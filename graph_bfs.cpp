// Foundational graph processing: adjacency-list representation, synthetic graph
// generation, and serial Breadth-First Search (BFS) baseline.
//
// This serial implementation serves as the correctness reference and performance
// baseline for future OpenMP and MPI parallelizations.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <vector>

// Graph Data Structure — Adjacency List 

struct Graph {
    int num_vertices;
    std::vector<std::vector<int>> adj; // adj[v] = list of neighbors of v

    explicit Graph(int V) : num_vertices(V), adj(V) {}
};

bool add_undirected_edge(Graph& graph, int u, int v) {
    if (u < 0 || v < 0 || u >= graph.num_vertices || v >= graph.num_vertices || u == v) {
        return false;
    }

    graph.adj[u].push_back(v);
    graph.adj[v].push_back(u);
    return true;
}

// Synthetic Undirected Graph Generator
// For every unique vertex pair (u, v) with u < v, an edge is added with
//
// Parameters:
//   V       — total number of vertices  (V >= 0)
//   density — edge probability in [0.0, 1.0]
//
// Complexity: O(V^2) — iterates over all possible edges.

Graph generate_graph(int V, double density, uint64_t seed = 42) {
    Graph g(V);

    // Clamp density to valid range
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

// Serial Breadth-First Search (BFS)
// Standard level-synchronous BFS that computes shortest-path distances
// (in number of hops) from a single source vertex.
//
// Parameters:
//   graph  — the adjacency-list graph
//   source — starting vertex (must be in [0, V))
//
// Returns:
//   std::vector<int> of size V where element i holds the shortest distance
//   from `source` to vertex i.  Unreachable vertices are marked with -1.
//
// Complexity: O(V + E)

std::vector<int> bfs_serial(const Graph& graph, int source) {
    const int V = graph.num_vertices;

    // Initialize all distances to -1 (unreachable)
    std::vector<int> distance(V, -1);

    // Frontier queue for BFS exploration
    std::queue<int> frontier;

    // Mark source
    distance[source] = 0;
    frontier.push(source);

    while (!frontier.empty()) {
        int current = frontier.front();
        frontier.pop();

        int next_dist = distance[current] + 1;

        // Explore all neighbors of the current vertex
        for (int neighbor : graph.adj[current]) {
            if (distance[neighbor] == -1) {       // not yet visited
                distance[neighbor] = next_dist;
                frontier.push(neighbor);
            }
        }
    }

    return distance;
}

// Output Helpers

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
        << "  --manual   Read graph data interactively from stdin\n"
        << "  --json     Output results as JSON (pipe to visualizer)\n"
        << "  --seed N   RNG seed for reproducibility  (default: 42)\n"
        << "  --help     Show this help message\n"
        << "\n"
        << "Examples:\n"
        << "  " << program << "                         # 10 nodes, 0.3 density\n"
        << "  " << program << " 100 0.05 0              # 100 nodes, sparse\n"
        << "  " << program << " --manual                 # prompt for V, source, and edges\n"
        << "  " << program << " --manual 6 0             # prompt only for the edge list\n"
        << "  " << program << " 50 0.2 3 --json          # JSON for visualizer\n"
        << "  " << program << " 20 0.4 0 --json > graph.json\n";
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
        int u = -1;
        int v = -1;

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

// Print human-readable results to stdout
void print_text(const Graph& graph, const std::vector<int>& dist, int source,
                double gen_ms, double bfs_ms) {
    const int V = graph.num_vertices;

    // Adjacency list (skip for large graphs)
    if (V <= 200) {
        std::cout << "\nAdjacency list:\n";
        for (int u = 0; u < V; ++u) {
            std::cout << "  " << u << " ->";
            for (std::size_t i = 0; i < graph.adj[u].size(); ++i) {
                std::cout << (i == 0 ? " " : ", ") << graph.adj[u][i];
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "\n(Adjacency list omitted for V > 200)\n";
    }

    // Distance array
    std::cout << "\nDistance array (hops from node " << source << "):\n";
    if (V <= 200) {
        for (int v = 0; v < V; ++v) {
            std::cout << "  node " << v << " : ";
            if (dist[v] == -1) std::cout << "unreachable";
            else                std::cout << dist[v];
            std::cout << "\n";
        }
    } else {
        // For large graphs, show a histogram of distances
        int max_dist = 0;
        for (int v = 0; v < V; ++v)
            if (dist[v] > max_dist) max_dist = dist[v];
        std::vector<int> hist(max_dist + 2, 0); // last bucket = unreachable
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

    // Summary
    int reachable = 0, total_edges = 0;
    for (int v = 0; v < V; ++v) {
        if (dist[v] != -1) ++reachable;
        total_edges += static_cast<int>(graph.adj[v].size());
    }
    total_edges /= 2;

    std::cout << "\nGraph stats: " << total_edges << " edges, "
              << reachable << "/" << V
              << " vertices reachable from node " << source << "\n";
    std::cout << "Timing: graph generation " << gen_ms
              << " ms, BFS " << bfs_ms << " ms\n";
}

// Emit a self-contained JSON object that the HTML visualizer can load
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

    // Edge list
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

    // Distance array
    std::cout << "  \"distance\": [";
    for (int v = 0; v < V; ++v) {
        if (v > 0) std::cout << ",";
        std::cout << dist[v];
    }
    std::cout << "]\n";
    std::cout << "}\n";
}

// Main — CLI entry point

int main(int argc, char* argv[]) {
    // --- Defaults ---
    int      V       = 10;
    double   density = 0.3;
    int      source  = 0;
    uint64_t seed    = 42;
    bool     json    = false;
    bool     manual  = false;

    // --- Parse arguments ---
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
        } else {
            positional.push_back(argv[i]);
        }
    }

    if (positional.size() >= 1) V       = std::stoi(positional[0]);
    if (manual) {
        if (positional.size() == 0) {
            if (!prompt_for_manual_config(V, source)) {
                return 1;
            }
        } else {
            if (positional.size() >= 1) V      = std::stoi(positional[0]);
            if (positional.size() >= 2) source = std::stoi(positional[1]);
        }
    } else {
        if (positional.size() >= 1) V       = std::stoi(positional[0]);
        if (positional.size() >= 2) density = std::stod(positional[1]);
        if (positional.size() >= 3) source  = std::stoi(positional[2]);
    }

    // --- Validate inputs ---
    if (V <= 0)      { std::cerr << "Error: V must be > 0\n"; return 1; }
    if (source < 0 || source >= V) {
        std::cerr << "Error: source must be in [0, " << V - 1 << "]\n";
        return 1;
    }
    if (!manual && (density < 0.0 || density > 1.0)) {
        std::cerr << "Error: density must be in [0.0, 1.0]\n";
        return 1;
    }

    Graph graph(V);

    // --- Graph Construction (timed) ---
    auto t0 = std::chrono::high_resolution_clock::now();
    if (manual) {
        if (!json) {
            std::cout << "Manual graph input mode: V=" << V
                      << ", source=" << source << "\n";
        }
        if (!read_manual_graph(graph)) {
            return 1;
        }
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

    // --- Serial BFS (timed) ---
    if (!json) std::cout << "Running serial BFS from source node " << source << " ...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<int> dist = bfs_serial(graph, source);
    auto t3 = std::chrono::high_resolution_clock::now();
    double bfs_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // --- Output ---
    if (json)
        print_json(graph, dist, source, gen_ms, bfs_ms);
    else
        print_text(graph, dist, source, gen_ms, bfs_ms);

    return 0;
}
