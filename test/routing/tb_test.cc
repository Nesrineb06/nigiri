#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/routing/tb/trex/bitfield_interner.h"
#include "nigiri/routing/tb/trex/bitfield_queue.h"
#include "nigiri/routing/tb/trex/bitfield_reached.h"
#include "nigiri/routing/tb/trex/customize.h"
#include "nigiri/routing/tb/trex/scan.h"
#include "nigiri/routing/tb/trex/transfer_search.h"
#include "nigiri/routing/tb/trex/trex_ibe.h"
#include "nigiri/routing/tb/trex/trex_partition.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;
using namespace nigiri::loader;
using namespace std::chrono_literals;
using namespace nigiri::test_data::hrd_timetable;

timetable load_gtfs(auto const& files) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2021_y / March / 1},
                    date::sys_days{2021_y / March / 8}};
  register_special_stations(tt);
  gtfs::load_timetable({}, source_idx_t{0}, files(), tt);
  finalize(tt);
  return tt;
}
timetable load_hrd(auto const& files) {
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  hrd::load_timetable(source_idx_t{0U}, loader::hrd::hrd_5_20_26, files(), tt);
  finalize(tt);
  return tt;
}

pareto_set<routing::journey> tripbased_search(timetable const& tt,
                                              tb::tb_data const& tbd,
                                              routing::query q) {
  static auto search_state = routing::search_state{};
  auto algo_state = tb::query_state{tt, tbd};

  return *(routing::search<direction::kForward, tb::query_engine<false>>{
      tt, nullptr, search_state, algo_state, std::move(q)}
               .execute()
               .journeys_);
}

pareto_set<routing::journey> tripbased_search(
    timetable const& tt,
    tb::tb_data const& tbd,
    std::string_view from,
    std::string_view to,
    routing::start_time_t const time) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({to, src}), 0_minutes, 0U}}};
  return tripbased_search(tt, tbd, std::move(q));
}

pareto_set<routing::journey> tripbased_intermodal_search(
    timetable const& tt,
    tb::tb_data const& tbd,
    std::vector<routing::offset> start,
    std::vector<routing::offset> destination,
    interval<unixtime_t> interval,
    std::uint8_t const min_connection_count = 0U,
    bool const extend_interval_earlier = false,
    bool const extend_interval_later = false) {
  auto q = routing::query{
      .start_time_ = interval,
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .start_ = std::move(start),
      .destination_ = std::move(destination),
      .min_connection_count_ = min_connection_count,
      .extend_interval_earlier_ = extend_interval_earlier,
      .extend_interval_later_ = extend_interval_later};
  return tripbased_search(tt, tbd, std::move(q));
}

std::string results_str(auto const& results, timetable const& tt) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n";
  }
  return ss.str();
}

mem_dir no_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, no_transfer) {
  auto const tt = load_gtfs(no_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  for (auto const transfers : tbd.segment_transfers_) {
    EXPECT_TRUE(transfers.empty());
  }
}

mem_dir same_day_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,06:00:00,06:00:00,S1,1,0,0
R1_MON,12:00:00,12:00:00,S1,0,0,0
R1_MON,13:00:00,13:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, same_day_transfer) {
  auto const tt = load_gtfs(same_day_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_EQ(1U, tbd.segment_transfers_[s].size());
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

// Positive case: a boundary-crossing transfer must get a rank above 0.
TEST(tb_customize, sets_rank_for_crossing_transfer) {
  auto tt = load_gtfs(same_day_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const from_segment = tbd.transport_first_segment_[transport_idx_t{0U}];

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});

  // s0 != s1 makes R0's hop an IBE; s1 != s2 makes the search leave the cell.
  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 1U;
  cell_ids[s1] = 2U;
  cell_ids[s2] = 3U;
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};

  ASSERT_EQ(0U, tbd.transfer_rank_[from_segment][0]);

  tb::customize(tt, tbd, partition);

  EXPECT_GE(tbd.transfer_rank_[from_segment][0], 1U);
}

// Negative case: a transfer inside one cell must keep rank 0.
TEST(tb_customize, keeps_rank_zero_for_transfer_within_cell) {
  auto tt = load_gtfs(same_day_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const from_segment = tbd.transport_first_segment_[transport_idx_t{0U}];

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});

  // All stops share one cell, so R0's hop is never an IBE.
  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 1U;
  cell_ids[s1] = 1U;
  cell_ids[s2] = 1U;
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};

  ASSERT_EQ(0U, tbd.transfer_rank_[from_segment][0]);

  tb::customize(tt, tbd, partition);

  EXPECT_EQ(0U, tbd.transfer_rank_[from_segment][0]);
}

// A rank must stop rising once its crossing merges away at a coarser level.
TEST(tb_customize, stops_rank_rise_when_crossing_merged_at_higher_level) {
  auto tt = load_gtfs(same_day_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const from_segment = tbd.transport_first_segment_[transport_idx_t{0U}];

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});

  // Level 0: cells 0/1/3 all differ. Level 1 (>>1): 0/0/1, so s0 and s1 merge.
  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 0U;
  cell_ids[s1] = 1U;
  cell_ids[s2] = 3U;
  auto const partition = tb::trex_partition{2, std::move(cell_ids)};

  ASSERT_EQ(0U, tbd.transfer_rank_[from_segment][0]);

  tb::customize(tt, tbd, partition);

  EXPECT_EQ(1U, tbd.transfer_rank_[from_segment][0]);
}

// Cells 0/2/4 stay distinct after >>1, so the rank climbs level by level to 2.
TEST(tb_customize, raises_rank_to_two_when_crossing_survives_higher_level) {
  auto tt = load_gtfs(same_day_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const from_segment = tbd.transport_first_segment_[transport_idx_t{0U}];

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});

  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 0U;
  cell_ids[s1] = 2U;
  cell_ids[s2] = 4U;
  auto const partition = tb::trex_partition{2, std::move(cell_ids)};

  ASSERT_EQ(0U, tbd.transfer_rank_[from_segment][0]);

  tb::customize(tt, tbd, partition);

  EXPECT_EQ(2U, tbd.transfer_rank_[from_segment][0]);
}

mem_dir two_independent_transfers_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
P0,P0,,,,,,
P1,P1,,,,,,
P2,P2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2
R2,DTA,R2,R2,"P0 -> P1",2
R3,DTA,R3,R3,"P1 -> P2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2
R2,MON,R2_MON,R2_MON,3
R3,MON,R3_MON,R3_MON,4

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,06:00:00,06:00:00,S1,1,0,0
R1_MON,12:00:00,12:00:00,S1,0,0,0
R1_MON,13:00:00,13:00:00,S2,1,0,0
R2_MON,00:00:00,00:00:00,P0,0,0,0
R2_MON,06:00:00,06:00:00,P1,1,0,0
R3_MON,12:00:00,12:00:00,P1,0,0,0
R3_MON,13:00:00,13:00:00,P2,1,0,0
)");
}

// Two independent transfers sharing no stops or routes must both get ranked.
TEST(tb_customize, ranks_multiple_independent_ibes_correctly) {
  auto tt = load_gtfs(two_independent_transfers_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const from_segment_s = tbd.transport_first_segment_[transport_idx_t{0U}];
  auto const from_segment_p = tbd.transport_first_segment_[transport_idx_t{2U}];

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});
  auto const p0 = tt.locations_.location_id_to_idx_.at({"P0", src});
  auto const p1 = tt.locations_.location_id_to_idx_.at({"P1", src});
  auto const p2 = tt.locations_.location_id_to_idx_.at({"P2", src});

  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 1U;
  cell_ids[s1] = 2U;
  cell_ids[s2] = 3U;
  cell_ids[p0] = 4U;
  cell_ids[p1] = 5U;
  cell_ids[p2] = 6U;
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};

  ASSERT_EQ(0U, tbd.transfer_rank_[from_segment_s][0]);
  ASSERT_EQ(0U, tbd.transfer_rank_[from_segment_p][0]);

  tb::customize(tt, tbd, partition);

  EXPECT_GE(tbd.transfer_rank_[from_segment_s][0], 1U);
  EXPECT_GE(tbd.transfer_rank_[from_segment_p][0], 1U);
}

// A/B invariant: the same query must return the same result after customize().
TEST(tb_customize, does_not_change_routing_result) {
  auto tt = load_gtfs(same_day_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});

  auto const departure = unixtime_t{sys_days{February / 28 / 2021} + 23h};
  auto const result_a = tripbased_search(tt, tbd, "S0", "S2", departure);

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});

  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 1U;
  cell_ids[s1] = 2U;
  cell_ids[s2] = 3U;
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};

  tb::customize(tt, tbd, partition);

  auto const result_b = tripbased_search(tt, tbd, "S0", "S2", departure);

  EXPECT_EQ(results_str(result_a, tt), results_str(result_b, tt));
}

// Same A/B invariant, on a fixture where customize() ranks two transfers.
TEST(tb_customize, does_not_change_routing_result_with_multiple_ibes) {
  auto tt = load_gtfs(two_independent_transfers_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});

  auto const departure = unixtime_t{sys_days{February / 28 / 2021} + 23h};
  auto const result_a_s = tripbased_search(tt, tbd, "S0", "S2", departure);
  auto const result_a_p = tripbased_search(tt, tbd, "P0", "P2", departure);

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});
  auto const p0 = tt.locations_.location_id_to_idx_.at({"P0", src});
  auto const p1 = tt.locations_.location_id_to_idx_.at({"P1", src});
  auto const p2 = tt.locations_.location_id_to_idx_.at({"P2", src});

  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 1U;
  cell_ids[s1] = 2U;
  cell_ids[s2] = 3U;
  cell_ids[p0] = 4U;
  cell_ids[p1] = 5U;
  cell_ids[p2] = 6U;
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};

  tb::customize(tt, tbd, partition);

  auto const result_b_s = tripbased_search(tt, tbd, "S0", "S2", departure);
  auto const result_b_p = tripbased_search(tt, tbd, "P0", "P2", departure);

  EXPECT_EQ(results_str(result_a_s, tt), results_str(result_b_s, tt));
  EXPECT_EQ(results_str(result_a_p, tt), results_str(result_b_p, tt));
}

constexpr auto const same_day_transfer_journeys = R"(
[2021-02-28 23:00, 2021-03-01 13:00]
TRANSFERS: 1
     FROM: (S0, S0) [2021-03-01 00:00]
       TO: (S2, S2) [2021-03-01 13:00]
leg 0: (S0, S0) [2021-03-01 00:00] -> (S1, S1) [2021-03-01 06:00]
   0: S0      S0..............................................                               d: 01.03 00:00 [01.03 00:00]  [{name=R0, day=2021-03-01, id=R0_MON, src=0}]
   1: S1      S1.............................................. a: 01.03 06:00 [01.03 06:00]
leg 1: (S1, S1) [2021-03-01 06:00] -> (S1, S1) [2021-03-01 06:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-01 12:00] -> (S2, S2) [2021-03-01 13:00]
   0: S1      S1..............................................                               d: 01.03 12:00 [01.03 12:00]  [{name=R1, day=2021-03-01, id=R1_MON, src=0}]
   1: S2      S2.............................................. a: 01.03 13:00 [01.03 13:00]
leg 3: (S2, S2) [2021-03-01 13:00] -> (S2, S2) [2021-03-01 13:00]
  FOOTPATH (duration=0)

)";

TEST(tb_query, same_day_transfer) {
  auto const tt = load_gtfs(same_day_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(std::string_view{same_day_transfer_journeys},
            results_str(results, tt));
}

mem_dir next_day_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,TUE,R1_TUE,R1_TUE,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R1_TUE,06:00:00,06:00:00,S1,0,0,0
R1_TUE,08:00:00,08:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, next_day_transfer) {
  auto const tt = load_gtfs(next_day_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(1, t.day_offset_);
}

// Nonzero day offset: R0 runs Monday, R1 Tuesday, so the days shift by one.
TEST(tb_customize, sets_rank_for_next_day_crossing_transfer) {
  auto tt = load_gtfs(next_day_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const from_segment = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_EQ(1U, tbd.segment_transfers_[from_segment].size());

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});

  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 1U;
  cell_ids[s1] = 2U;
  cell_ids[s2] = 3U;
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};

  ASSERT_EQ(0U, tbd.transfer_rank_[from_segment][0]);

  tb::customize(tt, tbd, partition);

  EXPECT_GE(tbd.transfer_rank_[from_segment][0], 1U);
}

// beta' = (beta & t.beta) << t.sigma: 1100000 & 100000 = 100000, then << 1.
TEST(tb_bitfield_queue, enqueue_shifts_day_bitfield_by_day_offset) {
  auto tt = load_gtfs(next_day_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const from_segment = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_EQ(1U, tbd.segment_transfers_[from_segment].size());
  auto const t = tbd.segment_transfers_[from_segment][0];
  ASSERT_EQ(1, t.get_day_offset());
  ASSERT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);

  auto const to_loc =
      stop{tt.route_location_seq_[t.route_][t.to_segment_offset_]}
          .location_idx();
  auto partition = tb::trex_partition{
      1, vector_map<location_idx_t, std::uint64_t>(tt.n_locations(),
                                                   std::uint64_t{0U})};
  auto const c = partition.get_cell_id_of_stop(to_loc, 0);

  auto reached = tb::bitfield_reached{tt.n_routes()};
  auto const bf_index = tb::bitfield_index{tbd};
  auto interner = tb::bitfield_interner{bf_index};
  auto q = tb::bitfield_queue{};
  q.q_.push_back(tb::bitfield_queue_entry{
      .segment_range_ = {from_segment, from_segment},
      .day_offset_ = 0,
      .days_ = interner.get_or_create(bitfield{"1100000"}),
      .parent_arrival_segment_ = from_segment,
      .parent_transfer_idx_ = 0U,
      .parent_ = tb::bitfield_queue_entry::kNoParent});

  auto const ok = q.enqueue(tt, tbd, partition, reached, interner, 0U,
                           from_segment, 0U, c, 0);

  ASSERT_TRUE(ok);
  ASSERT_EQ(2U, q.size());
  EXPECT_EQ(bitfield{"1000000"}, interner[q[1].days_]);
  EXPECT_EQ(std::int16_t{1}, q[1].day_offset_);
}

tb::bitfield_label reached_label(std::int16_t const day_offset,
                                 std::uint16_t const trip_offset,
                                 stop_idx_t const stop_idx,
                                 bitfield const& days) {
  return tb::bitfield_label{.day_offset_ = day_offset,
                            .trip_offset_ = trip_offset,
                            .stop_idx_ = stop_idx,
                            .days_ = days};
}

// Partial domination: 1100 and 1010 share one bit, so the candidate keeps 0010.
TEST(tb_bitfield_reached, shrinks_candidate_to_non_overlapping_days) {
  auto r = tb::bitfield_reached{1U};
  auto const route = route_idx_t{0U};

  r.update(route, reached_label(0, 0, stop_idx_t{0U}, bitfield{"1100"}));
  auto const result =
      r.update(route, reached_label(0, 0, stop_idx_t{0U}, bitfield{"1010"}));

  EXPECT_EQ(bitfield{"0010"}, result);
  ASSERT_EQ(1U, r.data_[route].size());
  EXPECT_EQ(bitfield{"1110"}, r.data_[route][0].days_);
}

// The merge must not swallow days already covered by the key.
TEST(tb_bitfield_reached, merges_equal_keys_without_losing_days) {
  auto r = tb::bitfield_reached{1U};
  auto const route = route_idx_t{0U};

  r.update(route, reached_label(3, 4, stop_idx_t{5U}, bitfield{"1000"}));
  r.update(route, reached_label(3, 4, stop_idx_t{5U}, bitfield{"0100"}));
  r.update(route, reached_label(3, 4, stop_idx_t{5U}, bitfield{"0010"}));

  ASSERT_EQ(1U, r.data_[route].size());
  EXPECT_EQ(bitfield{"1110"}, r.data_[route][0].days_);

  // every day already covered -> nothing survives, nothing is inserted
  EXPECT_EQ(bitfield{"0000"},
            r.update(route, reached_label(3, 4, stop_idx_t{5U},
                                          bitfield{"1110"})));
  EXPECT_EQ(1U, r.data_[route].size());

  // a fresh day still gets through and joins the same entry
  EXPECT_EQ(bitfield{"0001"},
            r.update(route, reached_label(3, 4, stop_idx_t{5U},
                                          bitfield{"1001"})));
  ASSERT_EQ(1U, r.data_[route].size());
  EXPECT_EQ(bitfield{"1111"}, r.data_[route][0].days_);
}

// Both dominated entries must be erased, including the one sliding into place.
TEST(tb_bitfield_reached, erases_every_fully_dominated_entry) {
  auto r = tb::bitfield_reached{1U};
  auto const route = route_idx_t{0U};

  r.update(route, reached_label(1, 1, stop_idx_t{1U}, bitfield{"1100"}));
  r.update(route, reached_label(2, 2, stop_idx_t{2U}, bitfield{"0011"}));
  ASSERT_EQ(2U, r.data_[route].size());

  auto const result =
      r.update(route, reached_label(0, 0, stop_idx_t{0U}, bitfield{"1111"}));

  EXPECT_EQ(bitfield{"1111"}, result);
  ASSERT_EQ(1U, r.data_[route].size());
  EXPECT_EQ(bitfield{"1111"}, r.data_[route][0].days_);
}

// A dominated entry that loses only some days must survive on the rest.
TEST(tb_bitfield_reached, shrinks_dominated_entry_that_survives) {
  auto r = tb::bitfield_reached{1U};
  auto const route = route_idx_t{0U};

  r.update(route, reached_label(1, 1, stop_idx_t{1U}, bitfield{"1100"}));
  auto const result =
      r.update(route, reached_label(0, 0, stop_idx_t{0U}, bitfield{"1000"}));

  EXPECT_EQ(bitfield{"1000"}, result);
  ASSERT_EQ(2U, r.data_[route].size());
  EXPECT_EQ(bitfield{"0100"}, r.data_[route][0].days_);
  EXPECT_EQ(bitfield{"1000"}, r.data_[route][1].days_);
}

// Non-comparable labels must keep all of their days.
TEST(tb_bitfield_reached, leaves_non_comparable_labels_untouched) {
  auto r = tb::bitfield_reached{1U};
  auto const route = route_idx_t{0U};

  r.update(route, reached_label(0, 5, stop_idx_t{0U}, bitfield{"1111"}));
  auto const result =
      r.update(route, reached_label(1, 0, stop_idx_t{0U}, bitfield{"1111"}));

  EXPECT_EQ(bitfield{"1111"}, result);
  ASSERT_EQ(2U, r.data_[route].size());
  EXPECT_EQ(bitfield{"1111"}, r.data_[route][0].days_);
  EXPECT_EQ(bitfield{"1111"}, r.data_[route][1].days_);
}

mem_dir long_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,76:00:00,76:00:00,S1,1,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, long_transfer) {
  auto const tt = load_gtfs(long_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(3, t.day_offset_);
}

mem_dir weekday_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WD,1,1,1,1,1,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,WD,R0_WD,R0_WD,1
R1,WD,R1_WD,R1_WD,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_WD,00:00:00,00:00:00,S0,0,0,0
R0_WD,02:00:00,02:00:00,S1,1,0,0
R1_WD,01:00:00,01:00:00,S1,0,0,0
R1_WD,03:00:00,03:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, weekday_transfer) {
  auto const tt = load_gtfs(weekday_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"0111100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(1, t.day_offset_);
}

mem_dir daily_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,DLY,R0_DLY,R0_DLY,1
R1,DLY,R1_DLY,R1_DLY,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_DLY,00:00:00,00:00:00,S0,0,0,0
R0_DLY,02:00:00,02:00:00,S1,1,0,0
R1_DLY,03:00:00,03:00:00,S1,0,0,0
R1_DLY,04:00:00,04:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, daily_transfer) {
  auto const tt = load_gtfs(daily_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"111111100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

mem_dir earlier_stop_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S4",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON0,R0_MON0,0
R0,MON,R0_MON1,R0_MON1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON0,00:00:00,00:00:00,S0,0,0,0
R0_MON0,01:00:00,01:00:00,S1,1,0,0
R0_MON0,02:00:00,02:00:00,S2,2,0,0
R0_MON0,03:00:00,03:00:00,S3,3,0,0
R0_MON0,04:00:00,04:00:00,S1,4,0,0
R0_MON0,05:00:00,05:00:00,S4,5,0,0
R0_MON1,04:00:00,04:00:00,S0,0,0,0
R0_MON1,05:00:00,05:00:00,S1,1,0,0
R0_MON1,06:00:00,06:00:00,S2,2,0,0
R0_MON1,07:00:00,07:00:00,S3,3,0,0
R0_MON1,08:00:00,08:00:00,S1,4,0,0
R0_MON1,09:00:00,09:00:00,S4,5,0,0
)");
}

TEST(tb_preprocess, earlier_stop_transfer) {
  auto const tt = load_gtfs(earlier_stop_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}] + 3U;
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}] + 1U,
            t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

constexpr auto const earlier_stop_transfer_journeys = R"(
[2021-02-28 23:00, 2021-03-01 06:00]
TRANSFERS: 1
     FROM: (S3, S3) [2021-03-01 03:00]
       TO: (S2, S2) [2021-03-01 06:00]
leg 0: (S3, S3) [2021-03-01 03:00] -> (S1, S1) [2021-03-01 04:00]
   3: S3      S3..............................................                               d: 01.03 03:00 [01.03 03:00]  [{name=R0, day=2021-03-01, id=R0_MON0, src=0}]
   4: S1      S1.............................................. a: 01.03 04:00 [01.03 04:00]
leg 1: (S1, S1) [2021-03-01 04:00] -> (S1, S1) [2021-03-01 04:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-01 05:00] -> (S2, S2) [2021-03-01 06:00]
   1: S1      S1..............................................                               d: 01.03 05:00 [01.03 05:00]  [{name=R0, day=2021-03-01, id=R0_MON1, src=0}]
   2: S2      S2.............................................. a: 01.03 06:00 [01.03 06:00]
leg 3: (S2, S2) [2021-03-01 06:00] -> (S2, S2) [2021-03-01 06:00]
  FOOTPATH (duration=0)

)";

TEST(tb_query, early_stop_transfer) {
  auto const tt = load_gtfs(earlier_stop_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_search(
      tt, tbd, "S3", "S2", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(std::string_view{earlier_stop_transfer_journeys},
            results_str(results, tt));
}

TEST(tb_query, no_journey_possible) {
  auto const tt = load_gtfs(earlier_stop_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_search(
      tt, tbd, "S4", "S0", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(0, results.size());
}

mem_dir earlier_transport_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S4",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON0,R0_MON0,0
R0,MON,R0_MON1,R0_MON1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON0,00:00:00,00:00:00,S0,0,0,0
R0_MON0,01:00:00,01:00:00,S1,1,0,0
R0_MON0,02:00:00,02:00:00,S2,2,0,0
R0_MON0,03:00:00,03:00:00,S3,3,0,0
R0_MON0,04:00:00,04:00:00,S1,4,0,0
R0_MON0,05:00:00,05:00:00,S4,5,0,0
R0_MON1,02:00:00,02:00:00,S0,0,0,0
R0_MON1,03:00:00,03:00:00,S1,1,0,0
R0_MON1,04:00:00,04:00:00,S2,2,0,0
R0_MON1,05:00:00,05:00:00,S3,3,0,0
R0_MON1,05:00:00,05:00:00,S1,4,0,0
R0_MON1,06:00:00,06:00:00,S4,5,0,0
)");
}

TEST(tb_preprocess, earlier_transport_transfer) {
  auto const tt = load_gtfs(earlier_transport_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{1U}];
  ASSERT_EQ(1U, tbd.segment_transfers_[s].size());
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{0U}] + 4U,
            t.to_segment_);
  // EXPECT_EQ(transport_idx_t{0U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

mem_dir uturn_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307


# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1 -> S2",2
R1,DTA,R1,R1,"S2 -> S1 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R1_MON,03:00:00,03:00:00,S2,0,0,0
R1_MON,04:00:00,04:00:00,S1,1,0,0
R1_MON,05:00:00,05:00:00,S3,2,0,0
)");
}

TEST(tb_preprocess, uturn_transfer) {
  auto const tt = load_gtfs(uturn_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  for (auto s = tb::segment_idx_t{1U}; s < tbd.segment_transfers_.size(); ++s) {
    EXPECT_TRUE(tbd.segment_transfers_[s].empty());
  }
  ASSERT_EQ(1U, tbd.segment_transfers_[tb::segment_idx_t{0U}].size());
  auto const t = tbd.segment_transfers_[tb::segment_idx_t{0U}][0];
  EXPECT_EQ(tb::segment_idx_t{3U}, t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

// scan() cuts at the first segment leaving the cell; earlier ones stay scannable.
TEST(tb_scan, truncates_at_first_segment_leaving_the_cell) {
  auto tt = load_gtfs(uturn_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const seg0 = tbd.transport_first_segment_[transport_idx_t{0U}];
  auto const seg1 = seg0 + tb::segment_idx_t{1U};

  auto const src = source_idx_t{0};
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});

  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s1] = 1U;
  cell_ids[s2] = 2U;
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};
  auto const c = std::uint64_t{1U};

  auto reached = tb::bitfield_reached{tt.n_routes()};
  auto const bf_index = tb::bitfield_index{tbd};
  auto interner = tb::bitfield_interner{bf_index};
  auto q = tb::bitfield_queue{};
  q.q_.push_back(tb::bitfield_queue_entry{
      .segment_range_ = {seg0, seg1 + tb::segment_idx_t{1U}},
      .day_offset_ = 0,
      .days_ = interner.get_or_create(bitfield{"100000"}),
      .parent_arrival_segment_ = seg0,
      .parent_transfer_idx_ = 0U,
      .parent_ = tb::bitfield_queue_entry::kNoParent});

  auto marked = std::vector<std::size_t>{};
  tb::scan(q, 0U, 1U, c, 0, tt, tbd, partition, reached, interner, marked);

  ASSERT_EQ(1U, marked.size());
  EXPECT_EQ(0U, marked[0]);  // queue index, not segment value
  EXPECT_EQ(seg0, q[0].segment_range_.from_);
  EXPECT_EQ(seg1, q[0].segment_range_.to_);
}

bool interned(tb::bitfield_interner const& interner, bitfield const& bf) {
  for (auto const& candidate : interner.local_) {
    if (candidate == bf) {
      return true;
    }
  }
  return false;
}

// End-to-end: searching the IBE must intern the computed day bitfield.
TEST(tb_customize, computes_expected_day_bitfield_end_to_end) {
  auto tt = load_gtfs(weekday_transfer_files);
  auto tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const from_segment = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_EQ(1U, tbd.segment_transfers_[from_segment].size());
  auto const t = tbd.segment_transfers_[from_segment][0];
  ASSERT_EQ(1, t.get_day_offset());

  auto const t_of_ibe = tbd.segment_transports_[from_segment];
  auto const seed_days = tt.bitfields_[tt.transport_traffic_days_[t_of_ibe]];
  auto const expected =
      bitfield{(seed_days & tbd.bitfields_[t.traffic_days_]) << 1};
  ASSERT_FALSE(expected.none());

  auto const src = source_idx_t{0};
  auto const s0 = tt.locations_.location_id_to_idx_.at({"S0", src});
  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", src});
  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", src});

  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  cell_ids[s0] = 1U;
  cell_ids[s1] = 2U;
  cell_ids[s2] = 3U;
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};

  auto const bf_index = tb::bitfield_index{tbd};
  ASSERT_EQ(nullptr, bf_index.find(expected));

  auto seeker = tb::transfer_search{};
  seeker.init(tt, tbd, partition, bf_index);
  seeker.run(from_segment, 0);

  EXPECT_TRUE(interned(seeker.interner_, expected));
}

// Same ranks no matter how the IBEs of a level are spread over threads.
TEST(tb_customize, parallel_result_matches_sequential) {
  auto tt = load_gtfs(two_independent_transfers_files);
  auto sequential = tb::preprocess(tt, profile_idx_t{0});
  auto parallel = tb::preprocess(tt, profile_idx_t{0});

  auto const src = source_idx_t{0};
  auto cell_ids = vector_map<location_idx_t, std::uint64_t>(
      tt.n_locations(), std::uint64_t{0U});
  auto cell = std::uint64_t{1U};
  for (auto const* id : {"S0", "S1", "S2", "P0", "P1", "P2"}) {
    cell_ids[tt.locations_.location_id_to_idx_.at({id, src})] = cell++;
  }
  auto const partition = tb::trex_partition{1, std::move(cell_ids)};

  tb::customize(tt, sequential, partition, 1U);
  tb::customize(tt, parallel, partition, 8U);

  auto const n_segments = sequential.transfer_rank_.size();
  ASSERT_EQ(n_segments, parallel.transfer_rank_.size());
  for (auto s = decltype(n_segments){0U}; s != n_segments; ++s) {
    auto const expected_ranks = sequential.transfer_rank_[tb::segment_idx_t{s}];
    auto const actual_ranks = parallel.transfer_rank_[tb::segment_idx_t{s}];
    ASSERT_EQ(expected_ranks.size(), actual_ranks.size());
    for (auto i = std::size_t{0U}; i != expected_ranks.size(); ++i) {
      EXPECT_EQ(expected_ranks[i], actual_ranks[i]);
    }
  }
}

mem_dir unnecessary_transfer_1_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1 -> S2",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R1_MON,01:00:00,01:10:00,S1,0,0,0
R1_MON,04:00:00,04:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, unnecessary_transfer_1) {
  auto const tt = load_gtfs(unnecessary_transfer_1_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  for (auto const transfers : tbd.segment_transfers_) {
    EXPECT_TRUE(transfers.empty());
  }
}

mem_dir unnecessary_transfer_2_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,
S5,S5,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2 -> S3 -> S4",2
R1,DTA,R1,R1,"S1 -> S2 -> S3 -> S5",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S2,1,0,0
R0_MON,02:00:00,02:00:00,S3,2,0,0
R0_MON,03:00:00,03:00:00,S4,3,0,0
R1_MON,00:10:00,00:10:00,S1,0,0,0
R1_MON,01:10:00,01:10:00,S2,1,0,0
R1_MON,02:10:00,02:10:00,S3,2,0,0
R1_MON,03:10:00,03:10:00,S5,3,0,0
)");
}

TEST(tb_preprocess, unnecessary_transfer_2) {
  auto const tt = load_gtfs(unnecessary_transfer_2_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}] + 1U;
  ASSERT_EQ(1U, tbd.segment_transfers_[s].size());
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}] + 2U,
            t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

mem_dir early_train_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2 -> S3",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,76:00:00,80:00:00,S2,1,0,0
R0_MON,81:00:00,81:00:00,S3,2,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}

constexpr auto const early_train_journeys = R"(
[2021-03-04 05:00, 2021-03-04 09:00]
TRANSFERS: 1
     FROM: (S1, S1) [2021-03-04 06:00]
       TO: (S3, S3) [2021-03-04 09:00]
leg 0: (S1, S1) [2021-03-04 06:00] -> (S2, S2) [2021-03-04 07:00]
   0: S1      S1..............................................                               d: 04.03 06:00 [04.03 06:00]  [{name=R1, day=2021-03-04, id=R1_THU, src=0}]
   1: S2      S2.............................................. a: 04.03 07:00 [04.03 07:00]
leg 1: (S2, S2) [2021-03-04 07:00] -> (S2, S2) [2021-03-04 07:02]
  FOOTPATH (duration=2)
leg 2: (S2, S2) [2021-03-04 08:00] -> (S3, S3) [2021-03-04 09:00]
   1: S2      S2..............................................                               d: 04.03 08:00 [04.03 08:00]  [{name=R0, day=2021-03-01, id=R0_MON, src=0}]
   2: S3      S3.............................................. a: 04.03 09:00 [04.03 09:00]
leg 3: (S3, S3) [2021-03-04 09:00] -> (S3, S3) [2021-03-04 09:00]
  FOOTPATH (duration=0)

)";

TEST(tb_preprocess, early_train_journeys) {
  auto const tt = load_gtfs(early_train_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  tbd.print(std::cout, tt);
  ASSERT_EQ("R1", tt.transport_name(transport_idx_t{1U}));
  // auto const s = tbd.transport_first_segment_[transport_idx_t{1U}];
  // ASSERT_EQ(1U, tbd.segment_transfers_[s].size());
}

TEST(tb_query, early_train) {
  auto const tt = load_gtfs(early_train_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_search(
      tt, tbd, "S1", "S3", unixtime_t{sys_days{March / 04 / 2021} + 5h});
  EXPECT_EQ(std::string_view{early_train_journeys}, results_str(results, tt));
}

constexpr auto const abc_journeys = R"(
[2020-03-30 05:00, 2020-03-30 07:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:15]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]
leg 3: (C, 0000003) [2020-03-30 07:15] -> (C, 0000003) [2020-03-30 07:15]
  FOOTPATH (duration=0)

)";

TEST(tb_query, abc) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results =
      tripbased_search(tt, tbd, "0000001", "0000003",
                       unixtime_t{sys_days{March / 30 / 2020} + 5h});
  EXPECT_EQ(std::string_view{abc_journeys}, results_str(results, tt));
}

constexpr auto const profile_abc_journeys = R"(
[2020-03-30 05:00, 2020-03-30 07:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:15]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]
leg 3: (C, 0000003) [2020-03-30 07:15] -> (C, 0000003) [2020-03-30 07:15]
  FOOTPATH (duration=0)

[2020-03-30 05:30, 2020-03-30 07:45]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:30]
       TO: (C, 0000003) [2020-03-30 07:45]
leg 0: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 1: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]
leg 3: (C, 0000003) [2020-03-30 07:45] -> (C, 0000003) [2020-03-30 07:45]
  FOOTPATH (duration=0)

)";

TEST(tb_query, profile_abc) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results =
      tripbased_search(tt, tbd, "0000001", "0000003",
                       interval{unixtime_t{sys_days{March / 30 / 2020}} + 5h,
                                unixtime_t{sys_days{March / 30 / 2020}} + 6h});
  EXPECT_EQ(std::string_view{profile_abc_journeys}, results_str(results, tt));
}

constexpr auto const intermodal_abc_journeys = R"(
[2020-03-30 05:20, 2020-03-30 08:00]
TRANSFERS: 1
     FROM: (START, START) [2020-03-30 05:20]
       TO: (END, END) [2020-03-30 08:00]
leg 0: (START, START) [2020-03-30 05:20] -> (A, 0000001) [2020-03-30 05:30]
  MUMO (id=99, duration=10)
leg 1: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 2: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 3: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]
leg 4: (C, 0000003) [2020-03-30 07:45] -> (END, END) [2020-03-30 08:00]
  MUMO (id=77, duration=15)

[2020-03-30 05:50, 2020-03-30 08:30]
TRANSFERS: 1
     FROM: (START, START) [2020-03-30 05:50]
       TO: (END, END) [2020-03-30 08:30]
leg 0: (START, START) [2020-03-30 05:50] -> (A, 0000001) [2020-03-30 06:00]
  MUMO (id=99, duration=10)
leg 1: (A, 0000001) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 07:00]
   0: 0000001 A...............................................                               d: 30.03 06:00 [30.03 08:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/360/0000002/420/, src=0}]
   1: 0000002 B............................................... a: 30.03 07:00 [30.03 09:00]
leg 2: (B, 0000002) [2020-03-30 07:00] -> (B, 0000002) [2020-03-30 07:02]
  FOOTPATH (duration=2)
leg 3: (B, 0000002) [2020-03-30 07:15] -> (C, 0000003) [2020-03-30 08:15]
   0: 0000002 B...............................................                               d: 30.03 07:15 [30.03 09:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/435/0000003/495/, src=0}]
   1: 0000003 C............................................... a: 30.03 08:15 [30.03 10:15]
leg 4: (C, 0000003) [2020-03-30 08:15] -> (END, END) [2020-03-30 08:30]
  MUMO (id=77, duration=15)

)";

TEST(tb_query, intermodal_abc) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_intermodal_search(
      tt, tbd,
      {{tt.locations_.location_id_to_idx_.at(
            {.id_ = "0000001", .src_ = source_idx_t{0U}}),
        10_minutes, 99U}},
      {{tt.locations_.location_id_to_idx_.at(
            {.id_ = "0000003", .src_ = source_idx_t{0U}}),
        15_minutes, 77U}},
      interval{unixtime_t{sys_days{March / 30 / 2020}} + 5_hours,
               unixtime_t{sys_days{March / 30 / 2020}} + 6_hours});
  EXPECT_EQ(std::string_view{intermodal_abc_journeys},
            results_str(results, tt));
}

