#include "nigiri/routing/tb/trex/bitfield_reached.h"

namespace nigiri::routing::tb {

bitfield_reached::bitfield_reached(std::size_t const n_routes)
    : data_{static_cast<std::uint32_t>(n_routes)} {}

void bitfield_reached::reset() {
  for (auto& v : data_) {
    v.clear();
  }
}

bitfield bitfield_reached::update(route_idx_t const route,
                                  bitfield_label label) {
  if (label.days_.none()) {
    return label.days_;
  }

  auto& entries = data_[route];
  for (auto i = std::size_t{0U}; i != entries.size();) {
    auto& other = entries[i];
    if (other.day_offset_ <= label.day_offset_ &&
        other.trip_offset_ <= label.trip_offset_ &&
        other.stop_idx_ <= label.stop_idx_) {
      // existing entry dominates -> shrink the new label's days
      label.days_ &= ~other.days_;
      if (label.days_.none()) {
        return label.days_;  // fully covered: do not insert
      }
    } else if (label.day_offset_ <= other.day_offset_ &&
               label.trip_offset_ <= other.trip_offset_ &&
               label.stop_idx_ <= other.stop_idx_) {
      // new label dominates -> shrink the existing entry's days
      other.days_ &= ~label.days_;
      if (other.days_.none()) {
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(i));
        continue;  // next element now at position i, don't advance
      }
    }
    ++i;
  }

  entries.push_back(label);
  return label.days_;
}

}  // namespace nigiri::routing::tb
