#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nigiri/types.h"

namespace nigiri::routing::tb {

// Label lambda = (sigma, rho, i, beta) for the bitfield Reached index R(L) (Algorithm 6).
struct bitfield_label {
  std::int16_t day_offset_;    // sigma: shift amount relative to the IBE
  std::uint16_t trip_offset_;  // rho: position within route_transport_ranges_[r]
  stop_idx_t stop_idx_;        // i: stop index
  bitfield days_;              // beta: days of the timetable this label is valid
};

// Per-route Reached index R
struct bitfield_reached {
  explicit bitfield_reached(std::size_t const n_routes)
      : data_{static_cast<std::uint32_t>(n_routes)} {}

  void reset() {
    for (auto& v : data_) {
      v.clear();
    }
  }

  // Algorithm 6 (Update): inserts label against R(route) per-day; returns the surviving bitfield.
  bitfield update(route_idx_t const route, bitfield_label label) {
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

  vector_map<route_idx_t, std::vector<bitfield_label>> data_;
};

}
