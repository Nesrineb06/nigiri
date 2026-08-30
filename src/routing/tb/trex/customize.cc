#include "nigiri/routing/tb/trex/customize.h"

#include <cstdint>
#include <limits>
#include <vector>

#include "utl/verify.h"

#include "nigiri/routing/tb/trex/bitfield_interner.h"
#include "nigiri/routing/tb/trex/bitfield_queue.h"
#include "nigiri/routing/tb/trex/bitfield_reached.h"
#include "nigiri/routing/tb/trex/scan.h"
#include "nigiri/routing/tb/trex/trex_ibe.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

namespace {

// Algorithm 4 lines 13-14
void unpack(bitfield_queue const& q,
           std::size_t index,
           std::uint8_t const rank,
           tb_data& tbd) {
  while (q[index].parent_ != bitfield_queue_entry::kNoParent) {
    tbd.transfer_rank_[q[index].parent_arrival_segment_]
                      [q[index].parent_transfer_idx_] = rank;
    index = q[index].parent_;
  }
}

}  // namespace

void customize(timetable const& tt,
               tb_data& tbd,
               trex_partition const& partition) {
  utl::verify(partition.number_of_levels_ > 0 &&
                  partition.number_of_levels_ <=
                      std::numeric_limits<std::uint8_t>::max(),
              "customize: invalid number of levels {}",
              partition.number_of_levels_);
  utl::verify(partition.cell_ids_.size() == tt.n_locations(),
              "customize: partition has {} cell ids, timetable has {} "
              "locations - partition and timetable do not match",
              partition.cell_ids_.size(), tt.n_locations());

  tbd.n_levels_ = static_cast<std::uint8_t>(partition.number_of_levels_);
  tbd.cell_ids_ = partition.cell_ids_;

  // r(t)=0 for all t already established by preprocess()'s zero-init of transfer_rank_.
  auto interner = bitfield_interner{tbd};
  auto ibes = collect_all_ibes_on_lowest_level(tt, tbd, partition);

  auto q = bitfield_queue{};
  q.q_.reserve(tbd.segment_transfers_.size());
  auto reached = bitfield_reached{tt.n_routes()};
  auto c_all = std::vector<std::size_t>{};
  auto marked = std::vector<std::size_t>{};

  for (auto l = 0; l != partition.number_of_levels_; ++l) {
    for (auto const ibe : ibes) {
      // c <- cell of T[i+1] on level l
      auto const t_of_ibe = tbd.segment_transports_[ibe];
      auto const route = tt.transport_route_[t_of_ibe];
      auto const pos_i = to_idx(ibe - tbd.transport_first_segment_[t_of_ibe]);
      auto const loc_i_plus_1 =
          stop{tt.route_location_seq_[route][pos_i + 1U]}.location_idx();
      auto const c = partition.get_cell_id_of_stop(loc_i_plus_1, l);

      q.reset();

      // Q1 <- {T[i+1,|T|,0,beta(T)}
      q.q_.push_back(bitfield_queue_entry{
          .segment_range_ = {ibe, tbd.get_segment_range(t_of_ibe).to_},
          .day_offset_ = 0,
          .days_ = interner.get_or_create(
              tt.bitfields_[tt.transport_traffic_days_[t_of_ibe]]),
          .parent_arrival_segment_ = ibe,
          .parent_transfer_idx_ = 0U,
          .parent_ = bitfield_queue_entry::kNoParent});

      // R / C <- reset (fresh per IBE).
      reached.reset();
      c_all.clear();

      // n <- 1; while Qn != empty do Scan(Qn, C, c, l); n <- n+1 (C accumulates across rounds, like Steil's toBeUnpacked).
      auto round_begin = std::size_t{0U};
      auto round_end = q.size();
      while (round_begin != round_end) {
        scan(q, round_begin, round_end, c, l, tt, tbd, partition, reached,
            interner, marked);
        c_all.insert(c_all.end(), std::make_move_iterator(marked.begin()),
                    std::make_move_iterator(marked.end()));
        round_begin = round_end;
        round_end = q.size();
      }

      // for each eps in C do: r(t) <- l+1 for all t in T[i]-to-eps subjourney
      for (auto const idx : c_all) {
        unpack(q, idx, static_cast<std::uint8_t>(l + 1), tbd);
      }
    }

    if (l + 1 != partition.number_of_levels_) {
      filter_irrelevant_ibes(ibes, tt, tbd, partition, l + 1);
    }
  }
}

}  // namespace nigiri::routing::tb
