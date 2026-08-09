#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "nigiri/routing/tb/bitfield_reached.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/routing/tb/trex_partition.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::tb {

// Trip segment T[j,k,sigma,beta] (Algorithm 5), used as a queue entry.
struct bitfield_queue_entry {
  static constexpr auto const kNoParent =
      std::numeric_limits<std::size_t>::max();

  interval<segment_idx_t> segment_range_;  // T[j, k]
  std::int16_t day_offset_;                // sigma
  bitfield days_;                          // beta
  segment_idx_t parent_arrival_segment_;   // segment the parent transfer left
  std::uint32_t parent_transfer_idx_;      // index into its transfer list
  std::size_t parent_;                     // index into bitfield_queue::q_
};

// Qn: the queue of pending trip segments.
struct bitfield_queue {
  void reset() { q_.clear(); }

  bitfield_queue_entry& operator[](std::size_t const pos) { return q_[pos]; }
  bitfield_queue_entry const& operator[](std::size_t const pos) const {
    return q_[pos];
  }

  std::size_t size() const { return q_.size(); }

  // Algorithm 7 (Enqueue)
  bool enqueue(timetable const& tt,
              tb_data const& tbd,
              trex_partition const& partition,
              bitfield_reached& reached,
              std::size_t const parent,
              segment_idx_t const from_segment,
              std::uint32_t const transfer_idx,
              std::uint64_t const c,
              int const l) {
    auto const t = tbd.segment_transfers_[from_segment][transfer_idx];

    if (tbd.transfer_rank_[from_segment][transfer_idx] < l) {
      return false;  // unimportant transfer
    }

    auto const to_loc =
        stop{tt.route_location_seq_[t.route_][t.to_segment_offset_]}
            .location_idx();
    if (partition.get_cell_id_of_stop(to_loc, l) != c) {
      return false;  // not in this cell
    }

    auto const& current = q_[parent];
    auto const sigma_prime = static_cast<std::int16_t>(
        current.day_offset_ + t.get_day_offset());

    // beta' = (beta & t.beta) << t.sigma
    auto beta_prime = current.days_ & tbd.bitfields_[t.traffic_days_];
    auto const shift = t.get_day_offset();
    beta_prime = shift < 0
                     ? (beta_prime >> static_cast<std::size_t>(-shift))
                     : (beta_prime << static_cast<std::size_t>(shift));

    beta_prime = reached.update(
        t.route_, bitfield_label{.day_offset_ = sigma_prime,
                                 .trip_offset_ = t.transport_offset_,
                                 .stop_idx_ = t.to_segment_offset_,
                                 .days_ = beta_prime});
    if (beta_prime.none()) {
      return false;  // trip segment already reached on all days
    }

    auto const t_prime =
        tt.route_transport_ranges_[t.route_].from_ + t.transport_offset_;
    q_.push_back(bitfield_queue_entry{
        .segment_range_ = {t.to_segment_, tbd.get_segment_range(t_prime).to_},
        .day_offset_ = sigma_prime,
        .days_ = beta_prime,
        .parent_arrival_segment_ = from_segment,
        .parent_transfer_idx_ = transfer_idx,
        .parent_ = parent});
    return true;
  }

  std::vector<bitfield_queue_entry> q_;
};

}  
