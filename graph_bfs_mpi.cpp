// Parallel BFS using MPI (level-synchronous, block-partitioned distributed-memory)

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>

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

// comp_ms / comm_ms — output parameters for timing breakdown
std::vector<int> bfs_mpi(const Graph& graph, int source,
                         int rank, int num_procs,
                         double& comp_ms, double& comm_ms) {
    const int V = graph.num_vertices;
    std::vector<int> distance(V, -1);

    int lo = static_cast<int>(static_cast<long long>(rank)     * V / num_procs);
    int hi = static_cast<int>(static_cast<long long>(rank + 1) * V / num_procs);

    distance[source] = 0;

    std::vector<int> current_frontier;
    current_frontier.push_back(source);

    int level = 0;
    comp_ms = 0.0;
    comm_ms = 0.0;

    while (!current_frontier.empty()) {
        level++;
        const int frontier_size = static_cast<int>(current_frontier.size());

        // ---- Computation phase ----
        double tc0 = MPI_Wtime();

        std::vector<int> local_next;

        for (int i = 0; i < frontier_size; ++i) {
            int current = current_frontier[i];
            for (int neighbor : graph.adj[current]) {
                if (neighbor >= lo && neighbor < hi && distance[neighbor] == -1) {
                    distance[neighbor] = level;
                    local_next.push_back(neighbor);
                }
            }
        }

        double tc1 = MPI_Wtime();
        comp_ms += (tc1 - tc0) * 1000.0;

        // ---- Communication phase ----
        double tm0 = MPI_Wtime();

        int local_count = static_cast<int>(local_next.size());
        std::vector<int> all_counts(num_procs);
        MPI_Allgather(&local_count, 1, MPI_INT,
                      all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        int total_next = 0;
        std::vector<int> displs(num_procs);
        for (int p = 0; p < num_procs; ++p) {
            displs[p] = total_next;
            total_next += all_counts[p];
        }

        if (total_next == 0) {
            double tm1 = MPI_Wtime();
            comm_ms += (tm1 - tm0) * 1000.0;
            break;
        }

        current_frontier.resize(total_next);
        MPI_Allgatherv(local_next.data(), local_count, MPI_INT,
                       current_frontier.data(), all_counts.data(),
                       displs.data(), MPI_INT, MPI_COMM_WORLD);

        double tm1 = MPI_Wtime();
        comm_ms += (tm1 - tm0) * 1000.0;

        // Propagate distances discovered by other ranks to this rank's array
        double tc2 = MPI_Wtime();
        for (int node : current_frontier) {
            if (distance[node] == -1)
                distance[node] = level;
        }
        double tc3 = MPI_Wtime();
        comp_ms += (tc3 - tc2) * 1000.0;
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

bool read_and_broadcast_manual_graph(Graph& graph, int rank) {
    int edge_count = 0;
    std::vector<int> edge_buf;  // flattened [u0, v0, u1, v1, ...]

    if (rank == 0) {
        std::cerr << "Enter number of undirected edges: ";
        if (!(std::cin >> edge_count) || edge_count < 0) {
            std::cerr << "Error: edge count must be a non-negative integer.\n";
            edge_count = -1;
        }
    }

    MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (edge_count < 0) return false;

    if (rank == 0) {
        std::cerr << "Enter edges as pairs: u v\n";
        std::cerr << "Example: 0 3\n";
        edge_buf.resize(edge_count * 2);

        for (int i = 0; i < edge_count; ++i) {
            int u = -1, v = -1;
            if (!(std::cin >> u >> v)) {
                std::cerr << "Error: failed to read edge " << i
                          << ". Expected two integers.\n";
                int err = -1;
                MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
                return false;
            }
            if (u < 0 || v < 0 || u >= graph.num_vertices ||
                v >= graph.num_vertices || u == v) {
                std::cerr << "Error: invalid edge (" << u << ", " << v << ")"
                          << ". Vertices must be distinct and in [0, "
                          << graph.num_vertices - 1 << "].\n";
                int err = -1;
                MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
                return false;
            }
            edge_buf[2 * i]     = u;
            edge_buf[2 * i + 1] = v;
        }
        int ok = 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int status = 0;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (status < 0) return false;
        edge_buf.resize(edge_count * 2);
    }

    MPI_Bcast(edge_buf.data(), edge_count * 2, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < edge_count; ++i)
        add_undirected_edge(graph, edge_buf[2 * i], edge_buf[2 * i + 1]);

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
        << "  --verify     Run serial BFS and compare for correctness\n"
        << "  --help       Show this help message\n"
        << "\n"
        << "MPI execution:\n"
        << "  mpiexec -n 4 " << program << " 1000 0.01 0\n"
        << "  mpiexec -n 4 " << program << " 1000 0.01 0 --verify\n"
        << "  mpiexec -n 2 " << program << " --manual\n";
}

void print_text(const Graph& graph, const std::vector<int>& dist, int source,
                double gen_ms, double bfs_ms, int num_procs,
                double comp_ms, double comm_ms, double serial_ms) {
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
    std::cout << "MPI processes: " << num_procs << "\n";
    std::cout << "Timing: graph generation " << gen_ms
              << " ms, MPI BFS " << bfs_ms << " ms\n";

    std::cout << "\n--- MPI Timing Breakdown ---\n";
    std::cout << "  Computation time: " << comp_ms << " ms\n";
    std::cout << "  Communication time: " << comm_ms << " ms\n";
    double comm_pct = (bfs_ms > 0.0) ? (comm_ms / bfs_ms * 100.0) : 0.0;
    std::cout << "  Communication overhead: " << comm_pct << "%\n";

    if (serial_ms > 0.0) {
        double speedup    = serial_ms / bfs_ms;
        double efficiency = speedup / num_procs;
        std::cout << "\n--- Performance Comparison ---\n";
        std::cout << "  Serial BFS time:   " << serial_ms << " ms\n";
        std::cout << "  MPI BFS time:      " << bfs_ms << " ms\n";
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
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int      V       = 10;
    double   density = 0.3;
    int      source  = 0;
    uint64_t seed    = 42;
    bool     json    = false;
    bool     manual  = false;
    bool     verify  = false;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (std::strcmp(argv[i], "--manual") == 0) {
            manual = true;
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else {
            positional.push_back(argv[i]);
        }
    }

    if (manual) {
        if (positional.size() == 0) {
            // rank 0 prompts; V and source are broadcast to all ranks
            int config[2] = {0, 0};
            if (rank == 0) {
                if (!prompt_for_manual_config(V, source)) {
                    config[0] = -1;
                    MPI_Bcast(config, 2, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Finalize();
                    return 1;
                }
                config[0] = V;
                config[1] = source;
            }
            MPI_Bcast(config, 2, MPI_INT, 0, MPI_COMM_WORLD);
            if (config[0] <= 0) { MPI_Finalize(); return 1; }
            V      = config[0];
            source = config[1];
        } else {
            if (positional.size() >= 1) V      = std::stoi(positional[0]);
            if (positional.size() >= 2) source = std::stoi(positional[1]);
        }
    } else {
        if (positional.size() >= 1) V       = std::stoi(positional[0]);
        if (positional.size() >= 2) density = std::stod(positional[1]);
        if (positional.size() >= 3) source  = std::stoi(positional[2]);
    }

    if (V <= 0) {
        if (rank == 0) std::cerr << "Error: V must be > 0\n";
        MPI_Finalize(); return 1;
    }
    if (source < 0 || source >= V) {
        if (rank == 0) std::cerr << "Error: source must be in [0, " << V - 1 << "]\n";
        MPI_Finalize(); return 1;
    }
    if (!manual && (density < 0.0 || density > 1.0)) {
        if (rank == 0) std::cerr << "Error: density must be in [0.0, 1.0]\n";
        MPI_Finalize(); return 1;
    }

    Graph graph(V);

    auto t0 = std::chrono::high_resolution_clock::now();
    if (manual) {
        if (rank == 0 && !json)
            std::cout << "Manual graph input mode: V=" << V
                      << ", source=" << source << "\n";
        if (!read_and_broadcast_manual_graph(graph, rank)) {
            MPI_Finalize();
            return 1;
        }
    } else {
        if (rank == 0 && !json)
            std::cout << "Generating undirected graph: V=" << V
                      << ", density=" << density
                      << ", source=" << source
                      << ", seed=" << seed << "\n";
        // all ranks use the same seed → identical graph on every process
        graph = generate_graph(V, density, seed);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double serial_ms = 0.0;
    std::vector<int> serial_dist;
    if (verify && rank == 0) {
        if (!json) std::cout << "Running serial BFS for verification ...\n";
        auto ts0 = std::chrono::high_resolution_clock::now();
        serial_dist = bfs_serial(graph, source);
        auto ts1 = std::chrono::high_resolution_clock::now();
        serial_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    }

    MPI_Barrier(MPI_COMM_WORLD);  // synchronize before timing

    if (rank == 0 && !json)
        std::cout << "Running MPI BFS (" << num_procs
                  << " processes) from source node " << source << " ...\n";

    double comp_ms = 0.0, comm_ms = 0.0;
    double t_start = MPI_Wtime();
    std::vector<int> dist = bfs_mpi(graph, source, rank, num_procs,
                                    comp_ms, comm_ms);
    double t_end = MPI_Wtime();
    // MPI_MAX gives the slowest rank's time — the true parallel wall time
    double bfs_local_ms = (t_end - t_start) * 1000.0;
    double bfs_ms = 0.0;
    MPI_Reduce(&bfs_local_ms, &bfs_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (verify && rank == 0) {
        bool correct = (dist == serial_dist);
        if (!json)
            std::cout << "\nCorrectness check: "
                      << (correct ? "PASSED (MPI == serial)"
                                  : "FAILED (mismatch detected!)")
                      << "\n";
        if (!correct) {
            int shown = 0;
            for (int v = 0; v < V && shown < 10; ++v) {
                if (dist[v] != serial_dist[v]) {
                    std::cerr << "  node " << v
                              << ": mpi=" << dist[v]
                              << " serial=" << serial_dist[v] << "\n";
                    ++shown;
                }
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        if (json)
            print_json(graph, dist, source, gen_ms, bfs_ms);
        else
            print_text(graph, dist, source, gen_ms, bfs_ms,
                       num_procs, comp_ms, comm_ms, serial_ms);
    }

    MPI_Finalize();
    return 0;
}
