#include "nigiri/routing/tb/trex/customize.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utl/verify.h"

#include "nigiri/routing/tb/trex/bitfield_interner.h"
#include "nigiri/routing/tb/trex/trex_ibe.h"
#include "nigiri/routing/tb/trex/transfer_search.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

namespace {

// Flat per-level IBE pool.
void run_level(std::vector<transfer_search>& seekers,
              std::vector<segment_idx_t> const& ibes,
              int const l) {
  auto const n = static_cast<std::ptrdiff_t>(ibes.size());

  if (seekers.size() == 1U) {
    // Keeps the one-thread baseline out of the OpenMP runtime.
    for (auto const ibe : ibes) {
      seekers[0].run(ibe, l);
    }
    return;
  }

#ifdef _OPENMP
  auto err = std::exception_ptr{};
  auto failed = std::atomic<bool>{false};

#pragma omp parallel num_threads(static_cast<int>(seekers.size()))
  {
    auto& seeker = seekers[static_cast<std::size_t>(omp_get_thread_num())];

#pragma omp for schedule(dynamic)
    for (auto i = std::ptrdiff_t{0}; i < n; ++i) {
      if (failed.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        seeker.run(ibes[static_cast<std::size_t>(i)], l);
      } catch (...) {
#pragma omp critical(trex_customize_error)
        {
          if (err == nullptr) {
            err = std::current_exception();
          }
        }
        failed.store(true, std::memory_order_relaxed);
      }
    }
  }

  if (err != nullptr) {
    std::rethrow_exception(err);
  }
#else
  for (auto i = std::ptrdiff_t{0}; i < n; ++i) {
    seekers[0].run(ibes[static_cast<std::size_t>(i)], l);
  }
#endif
}

}  // namespace

void customize(timetable const& tt,
               tb_data& tbd,
               trex_partition const& partition,
               unsigned n_threads) {
  utl::verify(partition.number_of_levels_ > 0 &&
                  partition.number_of_levels_ <=
                      std::numeric_limits<std::uint8_t>::max(),
              "customize: invalid number of levels {}",
              partition.number_of_levels_);
  utl::verify(partition.cell_ids_.size() == tt.n_locations(),
              "customize: partition has {} cell ids, timetable has {} "
              "locations - partition and timetable do not match",
              partition.cell_ids_.size(), tt.n_locations());

  tbd.n_levels_ = static_cast<std::uint8_t>(partition.number_of_levels_);
  tbd.cell_ids_ = partition.cell_ids_;

  // r(t)=0 for all t already established by preprocess()'s zero-init of transfer_rank_.
  auto const base = bitfield_index{tbd};
  auto ibes = collect_all_ibes_on_lowest_level(tt, tbd, partition);

  if (n_threads == 0U) {
#ifdef _OPENMP
    // n_threads == 0 -> ask OpenMP for its configured default thread count.
    n_threads = static_cast<unsigned>(std::max(1, omp_get_max_threads()));
#else
    n_threads = 1U;
#endif
  }
  auto const n_seekers =
      std::max(std::size_t{1U}, std::min(static_cast<std::size_t>(n_threads),
                                         ibes.size()));
  auto seekers = std::vector<transfer_search>(n_seekers);
  for (auto& seeker : seekers) {
    seeker.init(tt, tbd, partition, base);
  }

  for (auto l = 0; l != partition.number_of_levels_; ++l) {
    // Sequential
    run_level(seekers, ibes, l);

    if (l + 1 != partition.number_of_levels_) {
      filter_irrelevant_ibes(ibes, tt, tbd, partition, l + 1);
    }
  }
}

}  // namespace nigiri::routing::tb
