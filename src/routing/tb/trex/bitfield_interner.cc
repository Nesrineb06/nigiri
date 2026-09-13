#include "nigiri/routing/tb/trex/bitfield_interner.h"

#include <cstdint>

#include "utl/enumerate.h"
#include "utl/get_or_create.h"

namespace nigiri::routing::tb {

bitfield_index::bitfield_index(tb_data const& tbd)
    : bitfields_{&tbd.bitfields_} {
  for (auto const [idx, bf] : utl::enumerate(tbd.bitfields_)) {
    lookup_.emplace(bf, tb_bitfield_idx_t{idx});
  }
}

tb_bitfield_idx_t const* bitfield_index::find(bitfield const& bf) const {
  auto const it = lookup_.find(bf);
  return it == lookup_.end() ? nullptr : &it->second;
}

bitfield const& bitfield_index::operator[](tb_bitfield_idx_t const idx) const {
  return (*bitfields_)[idx];
}

std::size_t bitfield_index::size() const { return bitfields_->size(); }

bitfield_interner::bitfield_interner(bitfield_index const& base)
    : base_{&base} {}

bitfield const& bitfield_interner::operator[](
    tb_bitfield_idx_t const idx) const {
  auto const base_size = base_->size();
  return to_idx(idx) < base_size ? (*base_)[idx]
                                 : local_[to_idx(idx) - base_size];
}

tb_bitfield_idx_t bitfield_interner::get_or_create(bitfield const& bf) {
  if (auto const* existing = base_->find(bf); existing != nullptr) {
    return *existing;
  }
  return utl::get_or_create(lookup_, bf, [&]() {
    auto const idx = tb_bitfield_idx_t{
        static_cast<std::uint32_t>(base_->size() + local_.size())};
    local_.emplace_back(bf);
    return idx;
  });
}

}  // namespace nigiri::routing::tb
