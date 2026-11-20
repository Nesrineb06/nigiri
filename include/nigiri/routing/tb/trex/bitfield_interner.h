#pragma once

#include <cstddef>
#include <vector>

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

namespace nigiri::routing::tb {

// Read-only index of tb_data::bitfields_, built once and shared by all threads.
struct bitfield_index {
  explicit bitfield_index(tb_data const& tbd);

  // nullptr if bf is not in tb_data::bitfields_
  tb_bitfield_idx_t const* find(bitfield const& bf) const;

  bitfield const& operator[](tb_bitfield_idx_t idx) const;
  std::size_t size() const;

  vector_map<tb_bitfield_idx_t, bitfield> const* bitfields_;
  hash_map<bitfield, tb_bitfield_idx_t> lookup_;
};

// Thread-private: indices below base_->size() are shared, the rest are local_.
struct bitfield_interner {
  bitfield_interner() = default;
  explicit bitfield_interner(bitfield_index const& base);

  bitfield const& operator[](tb_bitfield_idx_t idx) const;
  tb_bitfield_idx_t get_or_create(bitfield const& bf);

  bitfield_index const* base_{nullptr};
  std::vector<bitfield> local_;
  hash_map<bitfield, tb_bitfield_idx_t> lookup_;
};

}  // namespace nigiri::routing::tb
