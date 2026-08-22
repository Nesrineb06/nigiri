#include "nigiri/routing/tb/trex/bitfield_queue.h"

#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

void bitfield_queue::reset() { q_.clear(); }

bitfield_queue_entry& bitfield_queue::operator[](std::size_t const pos) {
  return q_[pos];
}

bitfield_queue_entry const& bitfield_queue::operator[](
    std::size_t const pos) const {
  return q_[pos];
}

std::size_t bitfield_queue::size() const { return q_.size(); }

bool bitfield_queue::enqueue(timetable const& tt,
                             tb_data const& tbd,
                             trex_partition const& partition,
                             bitfield_reached& reached,
                             bitfield_interner& interner,
                             std::size_t const parent,
                             segment_idx_t const from_segment,
                             std::uint32_t const transfer_idx,
                             std::uint64_t const c,
                             int const l) {
  auto const t = tbd.segment_transfers_[from_segment][transfer_idx];

  if (tbd.transfer_rank_[from_segment][transfer_idx] < l) {
    return false;  // unimportant transfer
  }

  auto const to_loc =
      stop{tt.route_location_seq_[t.route_][t.to_segment_offset_]}
          .location_idx();
  if (partition.get_cell_id_of_stop(to_loc, l) != c) {
    return false;  // not in this cell
  }

  auto const& current = q_[parent];
  auto const sigma_prime = static_cast<std::int16_t>(
      current.day_offset_ + t.get_day_offset());

  // beta' = (beta & t.beta) << t.sigma
  auto beta_prime = tbd.bitfields_[current.days_] & tbd.bitfields_[t.traffic_days_];
  auto const shift = t.get_day_offset();
  beta_prime = shift < 0
                   ? (beta_prime >> static_cast<std::size_t>(-shift))
                   : (beta_prime << static_cast<std::size_t>(shift));

  beta_prime = reached.update(
      t.route_, bitfield_label{.day_offset_ = sigma_prime,
                               .trip_offset_ = t.transport_offset_,
                               .stop_idx_ = t.to_segment_offset_,
                               .days_ = beta_prime});
  if (beta_prime.none()) {
    return false;  // trip segment already reached on all days
  }

  auto const t_prime =
      tt.route_transport_ranges_[t.route_].from_ + t.transport_offset_;
  q_.push_back(bitfield_queue_entry{
      .segment_range_ = {t.to_segment_, tbd.get_segment_range(t_prime).to_},
      .day_offset_ = sigma_prime,
      .days_ = interner.get_or_create(beta_prime),
      .parent_arrival_segment_ = from_segment,
      .parent_transfer_idx_ = transfer_idx,
      .parent_ = parent});
  return true;
}

}  // namespace nigiri::routing::tb
