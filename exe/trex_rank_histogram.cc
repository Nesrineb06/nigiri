// Measures how the T-REX transfer-rank distribution changes as the timetable
// horizon grows.
//
// A transfer gets a nonzero rank if it is important on *any* day of the
// timetable, so the longer the horizon, the more transfers are expected to be
// pulled above rank zero -- and the less the query can prune. Steil evaluates
// T-REX on two consecutive weekdays only (his preprocessing scales linearly
// with the number of days), so the shape of that curve beyond a couple of days
// is unmeasured.
//
// Run in two steps, because the partition is computed by an external
// partitioner:
//
//   1) without --partition: loads the feed at the longest horizon, writes the
//      compact layout graph and its METIS export, and prints the partitioner
//      call to make.
//   2) with --partition: reads that partition back and prints, as CSV, the rank
//      histogram for every requested horizon.
//
// The partition is deliberately built once from the longest horizon and reused
// for every horizon: rebuilding it per horizon would change the layout graph
// and confound the measurement.

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <vector>

#include "boost/program_options.hpp"

#include "date/date.h"

#include "fmt/core.h"

#include "utl/progress_tracker.h"

#include "nigiri/common/parse_date.h"
#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/trex/customize.h"
#include "nigiri/routing/tb/trex/trex_partition.h"
#include "nigiri/timetable.h"

namespace fs = std::filesystem;
namespace bpo = boost::program_options;
using namespace nigiri;
using namespace nigiri::loader;
using namespace std::string_literals;

namespace {

timetable load_feed(fs::path const& in,
                    date::sys_days const start,
                    unsigned const n_days) {
  auto sources = std::vector<timetable_source>{
      {.tag_ = in.filename().generic_string(), .path_ = in.generic_string()}};
  return load(sources, finalize_options{},
              {start, start + date::days{n_days}});
}

}  // namespace

int main(int ac, char** av) {
  auto const progress_tracker = utl::activate_progress_tracker("trex-hist");
  auto const silencer = utl::global_progress_bars{true};

  auto in = fs::path{};
  auto graph_file = fs::path{"layout_graph.bin"};
  auto metis_file = fs::path{"layout_graph.metis"};
  auto partition_file = fs::path{};
  auto start_date = "2024-06-03"s;  // a Monday
  auto levels = 9;
  auto horizons = std::vector<unsigned>{2, 4, 7, 14, 21, 30, 60, 90};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("in,i", bpo::value(&in), "GTFS input path (directory or .zip)")  //
      ("start_date,s", bpo::value(&start_date)->default_value(start_date),
       "first day of the timetable, format: YYYY-MM-DD (pick a Monday)")  //
      ("levels,l", bpo::value(&levels)->default_value(levels),
       "number of partition levels")  //
      ("horizons", bpo::value(&horizons)->multitoken(),
       "timetable lengths in days to measure")  //
      ("graph", bpo::value(&graph_file)->default_value(graph_file),
       "compact layout graph file (written in step 1, read in step 2)")  //
      ("metis", bpo::value(&metis_file)->default_value(metis_file),
       "METIS export of the layout graph (written in step 1)")  //
      ("partition,p", bpo::value(&partition_file),
       "partitioner output; when given, runs the measurement");

  auto vm = bpo::variables_map{};
  bpo::store(bpo::command_line_parser(ac, av).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") != 0U || in.empty()) {
    std::cout << desc << "\n";
    return vm.count("help") != 0U ? 0 : 1;
  }

  if (horizons.empty()) {
    std::cerr << "no horizons given\n";
    return 1;
  }

  auto const start = parse_date(start_date);
  auto const longest =
      *std::max_element(begin(horizons), end(horizons));

  // ---- step 1: export the layout graph for the external partitioner --------
  if (partition_file.empty()) {
    fmt::println(std::clog, "loading {} for {} days to build the layout graph",
                 in.generic_string(), longest);
    auto const tt = load_feed(in, start, longest);
    auto const graph = routing::tb::create_compact_layout_graph(tt);

    routing::tb::write_compact_layout_graph(graph, graph_file);
    routing::tb::write_layout_graph_to_metis(graph, metis_file.generic_string());

    fmt::println(std::clog,
                 "\nwrote {} and {} ({} stops)\n\nnow run the partitioner, "
                 "e.g.\n  kaffpa {} --k={} --preconfiguration=strong "
                 "--output_filename=partition.txt\n\nthen re-run this with "
                 "--partition partition.txt",
                 graph_file.generic_string(), metis_file.generic_string(),
                 tt.n_locations(), metis_file.generic_string(), 1U << levels);
    return 0;
  }

  // ---- step 2: measure the rank distribution per horizon -------------------
  auto graph = routing::tb::read_compact_layout_graph(graph_file);
  auto const partition = routing::tb::read_partition_file(
      *graph, partition_file.generic_string(), levels);

  fmt::println("days,rank,transfers,percent");

  auto expected_n_locations = std::optional<std::uint32_t>{};
  for (auto const n_days : horizons) {
    auto const tt = load_feed(in, start, n_days);

    // The partition maps location_idx_t -> cell. If the number of locations
    // changes with the horizon, those indices no longer line up with the
    // partition built above and every cell lookup would be wrong.
    auto const n_locations = static_cast<std::uint32_t>(tt.n_locations());
    if (!expected_n_locations.has_value()) {
      expected_n_locations = n_locations;
    } else if (*expected_n_locations != n_locations) {
      fmt::println(std::cerr,
                   "location count changed between horizons ({} vs {}) -- the "
                   "partition no longer matches, aborting",
                   *expected_n_locations, n_locations);
      return 1;
    }

    auto tbd = routing::tb::preprocess(tt, profile_idx_t{0});
    routing::tb::customize(tt, tbd, partition);

    auto hist = std::map<std::uint8_t, std::size_t>{};
    auto total = std::size_t{0U};
    for (auto s = routing::tb::segment_idx_t{0U};
         s != tbd.transfer_rank_.size(); ++s) {
      for (auto const rank : tbd.transfer_rank_[s]) {
        ++hist[rank];
        ++total;
      }
    }

    for (auto const& [rank, n] : hist) {
      fmt::println("{},{},{},{:.2f}", n_days, rank, n,
                   total == 0U ? 0.0 : 100.0 * static_cast<double>(n) /
                                           static_cast<double>(total));
    }
    std::cout.flush();
  }
}
