#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/routing/tb/trex/bitfield_interner.h"
#include "nigiri/routing/tb/trex/bitfield_reached.h"
#include "nigiri/routing/tb/trex/trex_partition.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

// Trip segment T[j,k,sigma,beta] (Algorithm 5), used as a queue entry.
struct bitfield_queue_entry {
  static constexpr auto const kNoParent =
      std::numeric_limits<std::size_t>::max();

  interval<segment_idx_t> segment_range_;  // T[j, k]
  std::int16_t day_offset_;                // sigma
  tb_bitfield_idx_t days_;                 // beta, deduplicated via tb_data::bitfields_
  segment_idx_t parent_arrival_segment_;   // segment the parent transfer left
  std::uint32_t parent_transfer_idx_;      // index into its transfer list
  std::size_t parent_;                     // index into bitfield_queue::q_
};

// Qn: the queue of pending trip segments.
struct bitfield_queue {
  void reset();

  bitfield_queue_entry& operator[](std::size_t pos);
  bitfield_queue_entry const& operator[](std::size_t pos) const;

  std::size_t size() const;

  // Algorithm 7 (Enqueue)
  bool enqueue(timetable const& tt,
              tb_data const& tbd,
              trex_partition const& partition,
              bitfield_reached& reached,
              bitfield_interner& interner,
              std::size_t parent,
              segment_idx_t from_segment,
              std::uint32_t transfer_idx,
              std::uint64_t c,
              int l);

  std::vector<bitfield_queue_entry> q_;
};

}  // namespace nigiri::routing::tb
