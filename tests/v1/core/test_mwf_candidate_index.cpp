// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Standalone CPU checks for candidate ordering and rejected-operation state.

#include "mwf_candidate_index.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

using Index = vllm::MWFCandidateIndex;
using Kind = Index::Kind;
using Id = Index::BlockId;
static int checks = 0, groups = 0;

static void check(bool result, const std::string& message = "check failed") {
  ++checks;
  if (!result) throw std::runtime_error(message);
}

template <typename Error, typename Function>
static void rejects(Function action) {
  ++checks;
  try { action(); } catch (const Error&) { return; }
  throw std::runtime_error("expected exception was not raised");
}

static void pop(Index& index, std::initializer_list<Id> ids, double elapsed = 0) {
  check(index.pop_n(ids.size(), elapsed) == std::vector<Id>(ids));
}

static std::vector<Id> snapshot(Index index, double elapsed) {
  return index.pop_n(index.size(), elapsed);
}

template <typename Error, typename Function>
static void rejects_unchanged(Index& index, Function action,
                              double probe_elapsed = 4.0) {
  const auto size = index.size();
  const auto before = snapshot(index, 0);
  const auto later = snapshot(index, probe_elapsed);
  rejects<Error>(action);
  check(index.size() == size);
  check(snapshot(index, 0) == before);
  check(snapshot(index, probe_elapsed) == later);
}

struct ReferenceEntry {
  Kind kind;
  double a, b;
  unsigned order;
};

// Scan every candidate using its mathematical priority, without score indexes.
static std::vector<Id> scan_pop(std::map<Id, ReferenceEntry>& reference,
                                std::size_t n, double elapsed) {
  auto rank = [&](const auto& item) {
    const auto& entry = item.second;
    const int priority = entry.kind == Kind::Unhashed ? 0
                         : entry.kind == Kind::Missing ? 1 : 2;
    const double value = entry.kind == Kind::Tracked
        ? std::min(entry.a - elapsed / 1.25, entry.b - elapsed / 3.5)
        : entry.order;
    return std::tuple{priority, value, item.first};
  };
  std::vector<Id> result;
  for (std::size_t i = 0; i < n; ++i) {
    auto best = std::min_element(reference.begin(), reference.end(),
        [&](const auto& a, const auto& b) { return rank(a) < rank(b); });
    result.push_back(best->first);
    reference.erase(best);
  }
  return result;
}

template <typename Function>
static void group(const char* name, Function action) {
  const int before = checks;
  action();
  ++groups;
  std::cout << "PASS " << name << " (" << checks - before << " checks)\n";
}

int main() {
  try {
    group("elapsed changes the winning branch", [] {
      Index index(1, 2);
      for (double elapsed : {0.0, 4.0}) {
        index.add(0, Kind::Tracked, 0, 10);
        index.add(1, Kind::Tracked, 10, -1);
        if (elapsed == 0) pop(index, {1, 0}, elapsed);
        else pop(index, {0, 1}, elapsed);
        check(index.size() == 0);
      }
    });
    group("kind priority and free order", [] {
      Index index(1, 2);
      index.add(9, Kind::Tracked, -100, -100);
      index.add(8, Kind::Missing);
      index.add(7, Kind::Unhashed);
      index.add(6, Kind::Unhashed);
      index.add(5, Kind::Missing);
      pop(index, {7, 6, 8, 5, 9});
      index.add(9, Kind::Tracked, 0, 0);
      index.add(1, Kind::Missing);
      check(index.update(9, Kind::Missing));
      pop(index, {9, 1});
    });
    group("updates remove old keys and preserve unchanged keys", [] {
      Index index(1, 2);
      for (auto keys : {std::pair{-10.0, -20.0}, {-10.0, 20.0}, {10.0, -20.0}}) {
        index.add(0, Kind::Tracked, keys.first, keys.second);
        index.add(1, Kind::Tracked, 0, 0);
        check(index.update(0, Kind::Tracked, 10, 20));
        check(index.update(0, Kind::Tracked, 10, 20));
        pop(index, {1, 0});
      }
      index.add(0, Kind::Unhashed);
      check(index.update(0, Kind::Tracked, 1, 2));
      check(index.update(0, Kind::Unhashed));
      pop(index, {0});
    });
    group("erase removes both score branches", [] {
      Index index(1, 2);
      index.add(0, Kind::Tracked, -100, -200);
      index.add(1, Kind::Tracked, 0, 0);
      check(index.erase(0));
      check(!index.erase(0));
      check(!index.update(0, Kind::Tracked));
      check(index.size() == 1);
      pop(index, {1});
      check(index.size() == 0);
    });
    group("equal scores use block ID across branches", [] {
      Index index(1, 2);
      index.add(9, Kind::Tracked, 0, 2);
      index.add(3, Kind::Tracked, 2, 0);
      index.add(1, Kind::Tracked, 0, 0);
      pop(index, {1, 3, 9});
    });
    group("invalid inputs preserve candidates", [] {
      const double inf = std::numeric_limits<double>::infinity();
      const double nan = std::numeric_limits<double>::quiet_NaN();
      for (double bad : {0.0, -1.0, inf, -inf, nan}) {
        rejects<std::invalid_argument>([&] { Index index(bad, 1); });
        rejects<std::invalid_argument>([&] { Index index(1, bad); });
      }
      Index index(1, 2);
      index.add(0, Kind::Tracked, 0, 10);
      index.add(2, Kind::Tracked, 10, -1);
      index.add(3, Kind::Unhashed);
      index.add(4, Kind::Missing);
      auto unchanged = [&](auto action) {
        rejects_unchanged<std::invalid_argument>(index, action);
      };
      unchanged([&] { index.add(0, Kind::Unhashed); });
      unchanged([&] { index.add(-1, Kind::Unhashed); });
      for (int bad : {-1, 99}) {
        unchanged([&] { index.add(1, static_cast<Kind>(bad)); });
        unchanged([&] { index.update(0, static_cast<Kind>(bad)); });
      }
      for (auto keys : {std::pair{inf, 0.0}, {-inf, 0.0}, {nan, 0.0},
                        {0.0, inf}, {0.0, -inf}, {0.0, nan}}) {
        unchanged([&] { index.add(1, Kind::Tracked, keys.first, keys.second); });
        unchanged([&] { index.update(0, Kind::Tracked, keys.first, keys.second); });
      }
      unchanged([&] { index.pop_n(5, 0); });
      for (double bad : {-1.0, inf, -inf, nan})
        unchanged([&] { index.pop_n(2, bad); });
      check(index.pop_n(0, nan).empty());
      pop(index, {3, 4, 2, 0});
    });
    group("overflow rejects selection before removing candidates", [] {
      const double maximum = std::numeric_limits<double>::max();
      Index tiny(std::numeric_limits<double>::min(), 1);
      tiny.add(0, Kind::Tracked, 0, 0);
      tiny.add(7, Kind::Unhashed);
      rejects_unchanged<std::overflow_error>(
          tiny, [&] { tiny.pop_n(2, maximum); }, 0);
      pop(tiny, {7, 0});
      Index index(1, 1);
      index.add(0, Kind::Tracked, -maximum, -maximum);
      index.add(1, Kind::Missing);
      rejects_unchanged<std::overflow_error>(
          index, [&] { index.pop_n(2, maximum); }, 0);
      pop(index, {1, 0});
    });
    group("clear and logarithmic underflow ordering", [] {
      Index index(1, 2);
      index.add(1, Kind::Unhashed);
      index.add(2, Kind::Missing);
      index.add(3, Kind::Tracked, 1, 2);
      index.clear();
      check(index.size() == 0);
      check(index.pop_n(0, 0).empty());
      check(!index.erase(3));
      check(std::exp(-1000.0) == 0 && std::exp(-1001.0) == 0);
      index.add(1, Kind::Tracked, -1000, -1000);
      index.add(2, Kind::Tracked, -1001, -1001);
      pop(index, {2, 1});
    });
    group("10000 mixed operations vs scanning oracle (seed 314159)", [] {
      Index index(1.25, 3.5);
      std::map<Id, ReferenceEntry> reference;
      std::mt19937 random(314159);
      unsigned order = 0;
      double elapsed = 0;
      for (int step = 0; step < 10000; ++step) {
        elapsed += (random() % 31) / 8.0;
        Id id = random() % 64;
        ReferenceEntry next{static_cast<Kind>(random() % 3),
                            static_cast<double>(random() % 81) - 40,
                            static_cast<double>(random() % 81) - 40, order};
        const auto found = reference.find(id);
        const auto operation = random() % 100;
        if (operation < 42) {
          if (found != reference.end())
            rejects<std::invalid_argument>([&] { index.add(id, next.kind, next.a, next.b); });
          else {
            index.add(id, next.kind, next.a, next.b);
            next.order = order++;
            reference.emplace(id, next);
          }
        } else if (operation < 70) {
          check(index.update(id, next.kind, next.a, next.b) == (found != reference.end()));
          if (found != reference.end()) { next.order = found->second.order; found->second = next; }
        } else if (operation < 82) {
          check(index.erase(id) == (reference.erase(id) != 0));
        } else if (operation < 99) {
          const std::size_t n = random() %
              (std::min<std::size_t>(reference.size(), 4) + 2);
          if (n > reference.size())
            rejects<std::invalid_argument>([&] { index.pop_n(n, elapsed); });
          else {
            auto expected = scan_pop(reference, n, elapsed);
            check(index.pop_n(n, elapsed) == expected);
          }
        } else { index.clear(); reference.clear(); order = 0; elapsed = 0; }
        check(index.size() == reference.size());
        for (double probe : {0.0, elapsed, elapsed + 1000}) {
          auto copy = reference;
          const auto expected = scan_pop(copy, copy.size(), probe);
          check(snapshot(index, probe) == expected,
                "scanning oracle mismatch at seed 314159, step " +
                std::to_string(step) + ", elapsed " + std::to_string(probe));
        }
      }
    });
    std::cout << "PASS total " << groups << " groups, " << checks << " checks\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL after " << groups << " groups: " << error.what() << '\n';
    return 1;
  }
}
