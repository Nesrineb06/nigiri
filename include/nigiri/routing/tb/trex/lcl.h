#pragma once

#include <cstdint>
#include <vector>

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

namespace nigiri::routing::tb {

struct lcl_filter {
  void reset(tb_data const& tbd, bool enabled);

  bool enabled() const;

  // Every stop a journey of this query may start or end at.
  void add_anchor(location_idx_t l);

  // Minimum rank a transfer departing at `l` needs to be relevant.
  std::uint8_t min_rank(location_idx_t l) const;

  vector_map<location_idx_t, std::uint64_t> const* cell_ids_{nullptr};
  std::uint8_t n_levels_{0U};
  std::vector<std::uint64_t> anchors_;
};

}  // namespace nigiri::routing::tb
