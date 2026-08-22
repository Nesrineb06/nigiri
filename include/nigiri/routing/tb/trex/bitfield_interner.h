#pragma once

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

namespace nigiri::routing::tb {

// Deduplicates bitfields into tb_data::bitfields_
struct bitfield_interner {
  explicit bitfield_interner(tb_data& tbd);

  tb_bitfield_idx_t get_or_create(bitfield const& bf);

  tb_data& tbd_;
  hash_map<bitfield, tb_bitfield_idx_t> lookup_;
};

}  // namespace nigiri::routing::tb
