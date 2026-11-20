#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nigiri/types.h"

namespace nigiri::routing::tb {

inline bitfield shifted(bitfield const& b, int const s) {
  return s < 0 ? (b >> static_cast<std::size_t>(-s))
               : (b << static_cast<std::size_t>(s));
}

// Label lambda = (sigma, rho, i, beta) for the bitfield Reached index R(L) (Algorithm 6).
struct bitfield_label {
  std::int16_t day_offset_;    // sigma: shift amount relative to the IBE
  std::uint16_t trip_offset_;  // rho: position within route_transport_ranges_[r]
  stop_idx_t stop_idx_;        // i: stop index
  bitfield days_;              // beta: days of the timetable this label is valid

  bool precedes(bitfield_label const& o) const {
    return day_offset_ < o.day_offset_ ||
           (day_offset_ == o.day_offset_ && trip_offset_ <= o.trip_offset_);
  }
};

// Per-route Reached index R
struct bitfield_reached {
  bitfield_reached() = default;
  explicit bitfield_reached(std::size_t n_routes);

  // Clears only the routes touched since the last reset.
  void reset();

  // Algorithm 6 (Update): inserts label against R(route) per-day; returns the surviving bitfield.
  bitfield update(route_idx_t route, bitfield_label label);

  // Labels currently stored across all touched routes.
  std::size_t n_entries() const;

  std::uint64_t same_key_hits_{0U};
  std::uint64_t same_key_merges_{0U};

  vector_map<route_idx_t, std::vector<bitfield_label>> data_;

  std::vector<route_idx_t> touched_;
};

}  // namespace nigiri::routing::tb
