#include "nigiri/routing/tb/trex/lcl.h"

#include <algorithm>
#include <bit>

namespace nigiri::routing::tb {

void lcl_filter::reset(tb_data const& tbd, bool const enabled) {
  cell_ids_ = (enabled && tbd.is_customized()) ? &tbd.cell_ids_ : nullptr;
  n_levels_ = tbd.n_levels_;
  anchors_.clear();
}

bool lcl_filter::enabled() const {
  return cell_ids_ != nullptr && !anchors_.empty();
}

void lcl_filter::add_anchor(location_idx_t const l) {
  if (cell_ids_ == nullptr) {
    return;
  }
  auto const cell = (*cell_ids_)[l];
  if (std::find(begin(anchors_), end(anchors_), cell) == end(anchors_)) {
    anchors_.push_back(cell);
  }
}

std::uint8_t lcl_filter::min_rank(location_idx_t const l) const {
  if (!enabled()) {
    return 0U;
  }
  auto const cell = (*cell_ids_)[l];
  auto min = n_levels_;
  for (auto const anchor : anchors_) {
    min = std::min(min, static_cast<std::uint8_t>(std::bit_width(cell ^ anchor)));
    if (min == 0U) {
      break;
    }
  }
  return min;
}

}  // namespace nigiri::routing::tb
