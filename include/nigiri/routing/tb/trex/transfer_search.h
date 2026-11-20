#pragma once

#include <cstddef>
#include <vector>

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/routing/tb/trex/bitfield_interner.h"
#include "nigiri/routing/tb/trex/bitfield_queue.h"
#include "nigiri/routing/tb/trex/bitfield_reached.h"
#include "nigiri/routing/tb/trex/trex_partition.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

// Algorithm 4 lines 4-14 for one IBE; all mutable state is thread-private.
struct transfer_search {
  transfer_search() = default;

  void init(timetable const& tt,
            tb_data& tbd,
            trex_partition const& partition,
            bitfield_index const& base);

  bool initialized() const { return tt_ != nullptr; }

  // Raises r(t) to level+1 for every transfer on a subjourney leaving `level`.
  void run(segment_idx_t ibe, int level);

  timetable const* tt_{nullptr};
  tb_data* tbd_{nullptr};
  trex_partition const* partition_{nullptr};

  bitfield_interner interner_;
  bitfield_queue q_;
  bitfield_reached reached_;
  std::vector<std::size_t> c_all_;  // C: queue entries to unpack
  std::vector<std::size_t> marked_;  // scan() output of the current round
};

}  // namespace nigiri::routing::tb
