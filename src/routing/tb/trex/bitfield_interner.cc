#include "nigiri/routing/tb/trex/bitfield_interner.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"

namespace nigiri::routing::tb {

bitfield_interner::bitfield_interner(tb_data& tbd) : tbd_{tbd} {
  for (auto const [idx, bf] : utl::enumerate(tbd_.bitfields_)) {
    lookup_.emplace(bf, tb_bitfield_idx_t{idx});
  }
}

tb_bitfield_idx_t bitfield_interner::get_or_create(bitfield const& bf) {
  return utl::get_or_create(lookup_, bf, [&]() {
    auto const idx = tb_bitfield_idx_t{tbd_.bitfields_.size()};
    tbd_.bitfields_.emplace_back(bf);
    return idx;
  });
}

}  // namespace nigiri::routing::tb
