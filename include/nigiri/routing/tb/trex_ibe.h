#pragma once

#include <vector>

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/routing/tb/trex_partition.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

// IBEs at the finest level: one segment per boundary crossing
std::vector<segment_idx_t> collect_all_ibes_on_lowest_level(
    timetable const& tt, tb_data const& tbd, trex_partition const& partition);

// Shrinks `ibes` to crossings still relevant at `level`
void filter_irrelevant_ibes(std::vector<segment_idx_t>& ibes,
                            timetable const& tt,
                            tb_data const& tbd,
                            trex_partition const& partition,
                            int level);

}
