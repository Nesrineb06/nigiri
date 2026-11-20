#include "nigiri/routing/tb/trex/transfer_search.h"

#include <cstdint>
#include <iterator>

#include "nigiri/routing/tb/trex/scan.h"
#include "nigiri/routing/tb/trex/transfer_rank.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

namespace {

// Algorithm 4 lines 13-14
void unpack(bitfield_queue const& q,
           std::size_t index,
           std::uint8_t const rank,
           tb_data& tbd) {
  while (q[index].parent_ != bitfield_queue_entry::kNoParent) {
    store_rank(tbd, q[index].parent_arrival_segment_,
              q[index].parent_transfer_idx_, rank);
    index = q[index].parent_;
  }
}

}  // namespace

void transfer_search::init(timetable const& tt,
                          tb_data& tbd,
                          trex_partition const& partition,
                          bitfield_index const& base) {
  tt_ = &tt;
  tbd_ = &tbd;
  partition_ = &partition;
  interner_ = bitfield_interner{base};
  reached_ = bitfield_reached{tt.n_routes()};
}

void transfer_search::run(segment_idx_t const ibe, int const l) {
  auto const& tt = *tt_;
  auto& tbd = *tbd_;
  auto const& partition = *partition_;

  // c <- cell of T[i+1] on level l
  auto const t_of_ibe = tbd.segment_transports_[ibe];
  auto const route = tt.transport_route_[t_of_ibe];
  auto const pos_i = to_idx(ibe - tbd.transport_first_segment_[t_of_ibe]);
  auto const loc_i_plus_1 =
      stop{tt.route_location_seq_[route][pos_i + 1U]}.location_idx();
  auto const c = partition.get_cell_id_of_stop(loc_i_plus_1, l);

  q_.reset();

  // Q1 <- {T[i+1,|T|,0,beta(T)}
  q_.q_.push_back(bitfield_queue_entry{
      .segment_range_ = {ibe, tbd.get_segment_range(t_of_ibe).to_},
      .day_offset_ = 0,
      .days_ = interner_.get_or_create(
          tt.bitfields_[tt.transport_traffic_days_[t_of_ibe]]),
      .parent_arrival_segment_ = ibe,
      .parent_transfer_idx_ = 0U,
      .parent_ = bitfield_queue_entry::kNoParent});

  // R / C <- reset (fresh per IBE).
  reached_.reset();
  c_all_.clear();

  // while Qn != empty: Scan(Qn, C, c, l). C accumulates across rounds.
  auto round_begin = std::size_t{0U};
  auto round_end = q_.size();
  while (round_begin != round_end) {
    scan(q_, round_begin, round_end, c, l, tt, tbd, partition, reached_,
        interner_, marked_);
    c_all_.insert(c_all_.end(), std::make_move_iterator(marked_.begin()),
                 std::make_move_iterator(marked_.end()));
    round_begin = round_end;
    round_end = q_.size();
  }

  // for each eps in C do: r(t) <- l+1 for all t in T[i]-to-eps subjourney
  for (auto const idx : c_all_) {
    unpack(q_, idx, static_cast<std::uint8_t>(l + 1), tbd);
  }
}

}  // namespace nigiri::routing::tb
