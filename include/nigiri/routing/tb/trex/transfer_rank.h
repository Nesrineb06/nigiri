#pragma once

#include <cstdint>

#include "nigiri/routing/tb/tb_data.h"

namespace nigiri::routing::tb {

// Concurrent writers all store l+1, so only the byte access needs to be atomic.
std::uint8_t load_rank(tb_data const& tbd,
                       segment_idx_t segment,
                       std::uint32_t transfer_idx);

void store_rank(tb_data& tbd,
                segment_idx_t segment,
                std::uint32_t transfer_idx,
                std::uint8_t rank);

}  // namespace nigiri::routing::tb
