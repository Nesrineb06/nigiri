#include "nigiri/routing/tb/trex/scan.h"

#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

void scan(bitfield_queue& q,
         std::size_t const round_begin,
         std::size_t const round_end,
         std::uint64_t const c,
         int const l,
         timetable const& tt,
         tb_data const& tbd,
         trex_partition const& partition,
         bitfield_reached& reached,
         bitfield_interner& interner,
         std::vector<std::size_t>& marked) {
  marked.clear();

  // True if segment i's arrival stop (p(T[i])) lies in cell c at level l.
  auto const in_cell = [&](segment_idx_t const i) {
    auto const t_of_i = tbd.segment_transports_[i];
    auto const route = tt.transport_route_[t_of_i];
    auto const pos = to_idx(i - tbd.transport_first_segment_[t_of_i]);
    auto const loc =
        stop{tt.route_location_seq_[route][pos + 1U]}.location_idx();
    return partition.get_cell_id_of_stop(loc, l) == c;
  };

  // Marks + truncates at the first out-of-cell position
  for (auto idx = round_begin; idx != round_end; ++idx) {
    auto& entry = q[idx];
    for (auto i = entry.segment_range_.from_; i != entry.segment_range_.to_;
        ++i) {
      if (!in_cell(i)) {
        marked.push_back(idx);
        entry.segment_range_.to_ = i;
        break;
      }
    }
  }

  // Relaxes every transfer in range via enqueue()
  for (auto idx = round_begin; idx != round_end; ++idx) {
    auto const range = q[idx].segment_range_;
    for (auto i = range.from_; i != range.to_; ++i) {
      auto const n_transfers = tbd.segment_transfers_[i].size();
      for (auto transfer_idx = std::uint32_t{0U}; transfer_idx != n_transfers;
          ++transfer_idx) {
        q.enqueue(tt, tbd, partition, reached, interner, idx, i, transfer_idx,
                 c, l);
      }
    }
  }
}

}  // namespace nigiri::routing::tb
