#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/tb/queue.h"
#include "nigiri/routing/tb/reached.h"
#include "nigiri/routing/tb/settings.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

struct queue_entry;
struct segment_info;

struct reached_segment_event {
  segment_idx_t segment_;
  day_idx_t day_;
  std::uint8_t k_;
  unixtime_t arrival_time_;
};

struct query_state {
  static constexpr std::uint8_t kSegmentUnmarkedLevel{
      std::numeric_limits<std::uint8_t>::max()};

  query_state(timetable const& tt, tb_data const& tbd)
      : tbd_{tbd}, r_{tt, tbd} {
    t_min_.fill(unixtime_t::max());
    q_n_.q_.reserve(10'000'000);
    end_reachable_.resize(tbd.segment_transfers_.size());
    ibe_target_segments_.resize(tbd.segment_transfers_.size());
    segment_level_markers_.resize(tbd.segment_transfers_.size(),
                                  kSegmentUnmarkedLevel);
  }

  void reset() {
    utl::fill(parent_, queue_entry::kNoParent);
    r_.reset();
    q_n_.reset();
    end_reachable_.zero_out();
    reached_segment_events_.clear();
    ibe_target_segments_.zero_out();
    utl::fill(segment_level_markers_, kSegmentUnmarkedLevel);
  }

  tb_data const& tbd_;

  // avx_reached r_;
  // queue<avx_reached> q_n_{r_};

  reached r_;
  queue<reached> q_n_{r_};

  // minimum arrival times per round
  std::array<unixtime_t, kMaxTransfers + 1U> t_min_;
  std::array<queue_idx_t, kMaxTransfers + 1U> parent_;

  bitvec_map<segment_idx_t> end_reachable_;
  hash_map<segment_idx_t, duration_t> dist_to_dest_;
  std::vector<reached_segment_event> reached_segment_events_;
  bool collect_all_segments_{false};
  unixtime_t max_arrival_for_event_to_all_{unixtime_t::max()};
  bitvec_map<segment_idx_t> ibe_target_segments_;
  std::vector<std::uint8_t> segment_level_markers_;
};

struct query_stats {
  std::map<std::string, std::uint64_t> to_map() const {
    return {
        {"lower_bound_pruning", lower_bound_pruning_},
        {"n_segments_enqueued", n_segments_enqueued_},
        {"n_segments_pruned", n_segments_pruned_},
        {"n_enqueue_prevented_by_reached", n_enqueue_prevented_by_reached_},
        {"n_journeys_found", n_journeys_found_},
        {"n_rounds", n_rounds_},
        {"max_transfers_reached", max_transfers_reached_},
        {"max_pareto_set_size", max_pareto_set_size_},
    };
  }

  bool lower_bound_pruning_{false};
  std::uint64_t n_segments_enqueued_{0U};
  std::uint64_t n_segments_pruned_{0U};
  std::uint64_t n_enqueue_prevented_by_reached_{0U};
  std::uint64_t n_journeys_found_{0U};
  std::uint64_t n_rounds_{0U};
  std::uint64_t max_pareto_set_size_{0U};
  bool max_transfers_reached_{false};
};

template <bool UseLowerBounds>
struct query_engine {
  using algo_state_t = query_state;
  using algo_stats_t = query_stats;

  static constexpr bool kUseLowerBounds = UseLowerBounds;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  query_engine(timetable const&,
               rt_timetable const*,
               query_state&,
               bitvec const& is_dest,
               std::array<bitvec, kMaxVias> const&,
               std::vector<std::uint16_t> const& dist_to_dest,
               hash_map<location_idx_t, std::vector<td_offset>> const&,
               std::vector<std::uint16_t> const& lb,
               std::vector<via_stop> const&,
               day_idx_t base,
               clasz_mask_t,
               bool,
               bool,
               bool,
               transfer_time_settings);

  algo_stats_t get_stats() const { return stats_; }

  algo_state_t& get_state() { return state_; }

  void reset_arrivals() {
    state_.r_.reset();
    utl::fill(state_.t_min_, unixtime_t::max());
    utl::fill(state_.parent_, queue_entry::kNoParent);
    state_.reached_segment_events_.clear();
  }

  void next_start_time() {
    state_.q_n_.reset();
    state_.reached_segment_events_.clear();
  }

  void add_start(location_idx_t, unixtime_t);

  /** Enqueue one concrete departure event: trip @p t boards at stop @p i on
   *  traffic day @p day (same convention as @ref timetable::event_time).
   *  @returns false if the triple is invalid or out of @ref kTBMaxDayOffset range. */
  bool add_start_event(transport_idx_t t, stop_idx_t i, day_idx_t day);
  void set_collect_all_segments(bool enabled) {
    state_.collect_all_segments_ = enabled;
    if (enabled) {
      // Pure event-to-all PoC mode: clear destination-related state.
      state_.end_reachable_.zero_out();
      state_.dist_to_dest_.clear();
      state_.t_min_.fill(unixtime_t::max());
      state_.parent_.fill(queue_entry::kNoParent);
      utl::fill(state_.segment_level_markers_,
                query_state::kSegmentUnmarkedLevel);
    }
  }
  void set_ibe_target_segment(segment_idx_t const s, bool const enabled = true) {
    state_.ibe_target_segments_.set(s, enabled);
  }
  void set_max_arrival_for_event_to_all(unixtime_t t) {
    state_.max_arrival_for_event_to_all_ = t;
  }
  std::vector<std::uint8_t> const& segment_level_markers() const {
    return state_.segment_level_markers_;
  }
  std::vector<reached_segment_event> const& reached_segment_events() const {
    return state_.reached_segment_events_;
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const,
               pareto_set<journey>& results);

  void reconstruct(query const& q, journey& j) const;

private:
  void seg_dest(std::uint8_t k, queue_idx_t);
  void seg_prune(std::uint8_t k, queue_entry&);
  void seg_transfers(queue_idx_t, std::uint8_t k);
  void mark_path_to_segment(std::uint8_t k,
                            queue_idx_t q,
                            segment_idx_t target_segment);

  segment_info seg(segment_idx_t, queue_entry const&) const;
  segment_info seg(segment_idx_t, day_idx_t) const;

  timetable const& tt_;
  query_state& state_;
  bitvec const& is_dest_;
  std::vector<std::uint16_t> const& dist_to_dest_;
  std::vector<std::uint16_t> const& lb_;
  day_idx_t base_;
  query_stats stats_;
};

}  // namespace nigiri::routing::tb