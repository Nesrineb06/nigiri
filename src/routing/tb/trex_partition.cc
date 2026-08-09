#include "nigiri/routing/tb/trex_partition.h"

#include <fstream>

#include "cista/io.h"

#include "utl/verify.h"

#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

compact_layout_graph create_compact_layout_graph(timetable const& tt) {
  auto const n_locations =
      static_cast<std::uint32_t>(tt.locations_.coordinates_.size());
  auto uf = make_union_find(n_locations);

  // merge footpath-connected locations
  constexpr auto prf_idx = profile_idx_t{0};  // default profile
  for (auto l = location_idx_t{0U}; l < n_locations; ++l) {
    for (auto const& fp : tt.locations_.footpaths_out_[prf_idx][l]) {
      uf.unite(to_idx(l), to_idx(fp.target()));
    }
  }

  // node weight = number of locations merged into that group
  auto weight_of_nodes =
      vector_map<location_idx_t, std::uint32_t>(n_locations, 0U);
  for (auto l = location_idx_t{0U}; l < n_locations; ++l) {
    weight_of_nodes[location_idx_t{uf.find(to_idx(l))}] += 1U;
  }

  // edges between consecutive stops, weighted by transport count on the route
  auto edges = hash_map<std::pair<location_idx_t, location_idx_t>, std::uint32_t>{};
  for (auto r = route_idx_t{0U}; r < tt.route_location_seq_.size(); ++r) {
    auto const stop_seq = tt.route_location_seq_[r];
    auto const n_transports =
        static_cast<std::uint32_t>(tt.route_transport_ranges_[r].size());

    for (auto i = 1U; i < stop_seq.size(); ++i) {
      auto const from = location_idx_t{
          uf.find(to_idx(stop{stop_seq[i - 1]}.location_idx()))};
      auto const to = location_idx_t{
          uf.find(to_idx(stop{stop_seq[i]}.location_idx()))};

      if (from == to) continue;

      edges[{from, to}] += n_transports;
      edges[{to, from}] += n_transports;
    }
  }

  // assemble adjacency lists
  auto graph =
      compact_layout_graph{std::move(uf), std::move(weight_of_nodes), {}};
  graph.adjacency_.resize(n_locations);
  for (auto const& [key, weight] : edges) {
    graph.adjacency_[key.first].push_back(layout_edge{key.second, weight});
  }

  return graph;
}

void write_compact_layout_graph(compact_layout_graph const& graph,
                                std::filesystem::path const& path) {
  cista::write(path, graph);
}

cista::wrapped<compact_layout_graph> read_compact_layout_graph(
    std::filesystem::path const& path) {
  return cista::read<compact_layout_graph>(path);
}

void write_layout_graph_to_metis(compact_layout_graph const& graph,
                                 std::string const& file_name) {
  auto const n = graph.weight_of_nodes_.size();

  auto n_directed_edges = std::uint64_t{0U};
  for (auto const& adj : graph.adjacency_) {
    n_directed_edges += static_cast<std::uint64_t>(adj.size());
  }
  utl::verify(n_directed_edges % 2U == 0U,
              "compact layout graph must be undirected (odd edge count)");

  auto file = std::ofstream{file_name};
  utl::verify(file.is_open(), "could not open {} for writing", file_name);

  // n [vertices]  m [undirected edges]  f=11 [node- and edge-weighted]
  file << n << " " << (n_directed_edges / 2U) << " 11";

  // one line per vertex: "<node weight> <nbr1> <w1> <nbr2> <w2> ..."
  // e.g. "5 2 3 4 1" -> weight 5, edge to vertex 2 (weight 3), to vertex 4 (weight 1)
  for (auto l = location_idx_t{0U}; l != n; ++l) {
    file << "\n" << graph.weight_of_nodes_[l];
    for (auto const& e : graph.adjacency_[l]) {
      // METIS vertex ids are 1-based.
      file << " " << (to_idx(e.to_vertex_) + 1U) << " " << e.weight_;
    }
  }
}

trex_partition read_partition_file(compact_layout_graph& graph,
                                   std::string const& file_name,
                                   int const num_levels) {
  auto const n = graph.weight_of_nodes_.size();

  auto file = std::ifstream{file_name};
  utl::verify(file.is_open(), "could not open {} for reading", file_name);

  auto global_ids = std::vector<std::uint64_t>(n);
  for (auto i = std::uint32_t{0U}; i != n; ++i) {
    utl::verify(static_cast<bool>(file >> global_ids[i]),
                "partition file {} has fewer than {} entries", file_name, n);
  }

  // expand back from layout-graph vertex (union-find root) to every merged location
  auto partition = trex_partition{num_levels, {}};
  partition.cell_ids_.resize(n);
  for (auto l = location_idx_t{0U}; l != n; ++l) {
    partition.cell_ids_[l] = global_ids[graph.union_find_.find(to_idx(l))];
  }

  return partition;
}

void write_trex_partition(trex_partition const& partition,
                          std::filesystem::path const& path) {
  cista::write(path, partition);
}

cista::wrapped<trex_partition> read_trex_partition(
    std::filesystem::path const& path) {
  return cista::read<trex_partition>(path);
}

}
