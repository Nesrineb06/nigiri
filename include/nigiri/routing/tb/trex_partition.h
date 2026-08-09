#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "cista/memory_holder.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

// Union-find with union-by-rank + path compression.
struct union_find {
  inline void clear() noexcept {
    for (auto& p : parent_) {
      p = n_;
    }
  }

  inline std::uint32_t find(std::uint32_t const i) noexcept {
    if (parent_[i] >= n_) {
      return i;
    } else {
      parent_[i] = find(parent_[i]);
      return parent_[i];
    }
  }

  inline std::uint32_t operator()(std::uint32_t const i) noexcept {
    return find(i);
  }

  inline void unite(std::uint32_t const i, std::uint32_t const j) noexcept {
    if (find(i) != find(j)) {
      link(find(i), find(j));
    }
  }

  inline void operator()(std::uint32_t const i, std::uint32_t const j) noexcept {
    unite(i, j);
  }

protected:
  inline void link(std::uint32_t const i, std::uint32_t const j) noexcept {
    if (parent_[i] < parent_[j]) {
      parent_[i] = j;
    } else if (parent_[j] < parent_[i]) {
      parent_[j] = i;
    } else {
      parent_[i] = j;
      ++parent_[j];
    }
  }

public:
  vector<std::uint32_t> parent_;
  std::uint32_t n_{0U};
};

inline union_find make_union_find(std::uint32_t const n) {
  auto parent = vector<std::uint32_t>{};
  parent.resize(n, n);
  return union_find{std::move(parent), n};
}

struct layout_edge {
  location_idx_t to_vertex_;
  std::uint32_t weight_;
};

struct compact_layout_graph {
  union_find union_find_;
  vector_map<location_idx_t, std::uint32_t> weight_of_nodes_;
  vecvec<location_idx_t, layout_edge> adjacency_;
};

compact_layout_graph create_compact_layout_graph(timetable const& tt);

void write_compact_layout_graph(compact_layout_graph const& graph,
                                std::filesystem::path const& path);
cista::wrapped<compact_layout_graph> read_compact_layout_graph(
    std::filesystem::path const& path);

void write_layout_graph_to_metis(compact_layout_graph const& graph,
                                 std::string const& file_name);


struct trex_partition {
  std::uint64_t get_cell_id_of_stop(location_idx_t const l,
                                    int const level) const {
    return cell_ids_[l] >> level;
  }

  bool in_same_cell(location_idx_t const a,
                    location_idx_t const b,
                    int const level) const {
    return get_cell_id_of_stop(a, level) == get_cell_id_of_stop(b, level);
  }

  int number_of_levels_{0};
  vector_map<location_idx_t, std::uint64_t> cell_ids_;
};


trex_partition read_partition_file(compact_layout_graph& graph,
                                   std::string const& file_name,
                                   int num_levels);

void write_trex_partition(trex_partition const& partition,
                          std::filesystem::path const& path);
cista::wrapped<trex_partition> read_trex_partition(
    std::filesystem::path const& path);

}
