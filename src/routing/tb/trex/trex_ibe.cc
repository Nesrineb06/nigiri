#include "nigiri/routing/tb/trex/trex_ibe.h"

#include <algorithm>

#include "utl/verify.h"

#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

std::vector<segment_idx_t> collect_all_ibes_on_lowest_level(
    timetable const& tt, tb_data const& tbd, trex_partition const& partition) {
  auto ibes = std::vector<segment_idx_t>{};

  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const stop_seq = tt.route_location_seq_[r];
    for (auto i = 0U; i + 1U < stop_seq.size(); ++i) {
      auto const from = stop{stop_seq[i]}.location_idx();
      auto const to = stop{stop_seq[i + 1U]}.location_idx();

      if (!partition.in_same_cell(from, to, 0)) {
        for (auto const t : tt.route_transport_ranges_[r]) {
          ibes.push_back(tbd.transport_first_segment_[t] + segment_idx_t{i});
        }
      }
    }
  }

  return ibes;
}

void filter_irrelevant_ibes(std::vector<segment_idx_t>& ibes,
                            timetable const& tt,
                            tb_data const& tbd,
                            trex_partition const& partition,
                            int const level) {
  utl::verify(level >= 0 && level < partition.number_of_levels_,
              "filter_irrelevant_ibes: level {} out of range [0, {})", level,
              partition.number_of_levels_);

  auto const crosses_boundary_at_level = [&](segment_idx_t const s) {
    auto const t = tbd.segment_transports_[s]; //which transport
    auto const r = tt.transport_route_[t]; //which route 
    auto const i = to_idx(s - tbd.transport_first_segment_[t]); //which stop index
    auto const stop_seq = tt.route_location_seq_[r];
    auto const from = stop{stop_seq[i]}.location_idx();
    auto const to = stop{stop_seq[i + 1U]}.location_idx();
    return !partition.in_same_cell(from, to, level);
  };

  ibes.erase(std::remove_if(ibes.begin(), ibes.end(),
                            [&](segment_idx_t const s) {
                              return !crosses_boundary_at_level(s);
                            }),
             ibes.end());
}

}
