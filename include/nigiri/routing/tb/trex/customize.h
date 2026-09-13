#pragma once

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/routing/tb/trex/trex_partition.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

// Algorithm 4: Customize. IBEs of a level run in parallel (0 = all cores).
void customize(timetable const& tt,
               tb_data& tbd,
               trex_partition const& partition,
               unsigned n_threads = 0U);

}  // namespace nigiri::routing::tb
