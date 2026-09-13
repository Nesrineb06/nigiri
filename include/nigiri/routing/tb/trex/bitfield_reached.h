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
  bitfield_reached() = default;
  explicit bitfield_reached(std::size_t n_routes);

  void reset();

  // Algorithm 6 (Update): inserts label against R(route) per-day; returns the surviving bitfield.
  bitfield update(route_idx_t route, bitfield_label label);

  vector_map<route_idx_t, std::vector<bitfield_label>> data_;
};

}  // namespace nigiri::routing::tb
