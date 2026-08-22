#pragma once

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/routing/tb/trex/trex_partition.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

// Algorithm 4: Customize.
void customize(timetable const& tt,
               tb_data& tbd,
               trex_partition const& partition);

}  // namespace nigiri::routing::tb
