#pragma once

#include <cstdint>
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

// Algorithm 5 (Scan)
void scan(bitfield_queue& q,
         std::size_t round_begin,
         std::size_t round_end,
         std::uint64_t c,
         int l,
         timetable const& tt,
         tb_data const& tbd,
         trex_partition const& partition,
         bitfield_reached& reached,
         bitfield_interner& interner,
         std::vector<std::size_t>& marked);

}  // namespace nigiri::routing::tb
