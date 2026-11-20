#include "nigiri/routing/tb/trex/transfer_rank.h"

#include <atomic>

namespace nigiri::routing::tb {

std::uint8_t load_rank(tb_data const& tbd,
                       segment_idx_t const segment,
                       std::uint32_t const transfer_idx) {
  // tbd is never a const object during customize(), only this view is const.
  auto& slot =
      const_cast<std::uint8_t&>(tbd.transfer_rank_[segment][transfer_idx]);
  return std::atomic_ref<std::uint8_t>{slot}.load(std::memory_order_relaxed);
}

void store_rank(tb_data& tbd,
                segment_idx_t const segment,
                std::uint32_t const transfer_idx,
                std::uint8_t const rank) {
  std::atomic_ref<std::uint8_t>{tbd.transfer_rank_[segment][transfer_idx]}
      .store(rank, std::memory_order_relaxed);
}

}  // namespace nigiri::routing::tb
