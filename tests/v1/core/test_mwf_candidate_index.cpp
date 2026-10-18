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
#include <tuple>

using Index = vllm::MWFCandidateIndex;
using Kind = Index::Kind;
using Id = Index::BlockId;
static int checks = 0, groups = 0;

static void check(bool result) {
  ++checks;
  if (!result) throw std::runtime_error("check failed");
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
      for (double bad : {0.0, -1.0, inf, nan}) {
        rejects<std::invalid_argument>([&] { Index index(bad, 1); });
        rejects<std::invalid_argument>([&] { Index index(1, bad); });
      }
      Index index(1, 2);
      index.add(0, Kind::Tracked, 0, 0);
      index.add(2, Kind::Tracked, 1, 1);
      rejects<std::invalid_argument>([&] { index.add(0, Kind::Tracked, 1, 1); });
      rejects<std::invalid_argument>([&] { index.add(-1, Kind::Unhashed); });
      rejects<std::invalid_argument>([&] { index.add(1, static_cast<Kind>(99)); });
      rejects<std::invalid_argument>([&] { index.update(0, static_cast<Kind>(99)); });
      for (auto keys : {std::pair{inf, 0.0}, {0.0, nan}}) {
        rejects<std::invalid_argument>([&] { index.add(1, Kind::Tracked, keys.first, keys.second); });
        rejects<std::invalid_argument>([&] { index.update(0, Kind::Tracked, keys.first, keys.second); });
      }
      rejects<std::invalid_argument>([&] { index.pop_n(3, 0); });
      for (double bad : {-1.0, inf, nan})
        rejects<std::invalid_argument>([&] { index.pop_n(1, bad); });
      check(index.size() == 2);
      pop(index, {0, 2});
    });
    group("overflow rejects selection before removing candidates", [] {
      const double maximum = std::numeric_limits<double>::max();
      Index tiny(std::numeric_limits<double>::min(), 1);
      tiny.add(0, Kind::Tracked, 0, 0);
      rejects<std::overflow_error>([&] { tiny.pop_n(1, maximum); });
      check(tiny.size() == 1);
      pop(tiny, {0});
      Index index(1, 1);
      index.add(0, Kind::Tracked, -maximum, -maximum);
      rejects<std::overflow_error>([&] { index.pop_n(1, maximum); });
      check(index.size() == 1);
      pop(index, {0});
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
    group("2000 seeded operations against a scanning reference", [] {
      struct Entry { Kind kind; double a, b; unsigned order; };
      Index index(1.25, 3.5);
      std::map<Id, Entry> reference;
      std::mt19937 random(314159);
      unsigned order = 0;
      for (int step = 0; step < 2000; ++step) {
        Id id = random() % 32;
        Entry next{static_cast<Kind>(random() % 3),
                   static_cast<double>(random() % 81) - 40,
                   static_cast<double>(random() % 81) - 40, order};
        const auto found = reference.find(id);
        const auto operation = random() % 8;
        if (operation < 3) {
          if (found != reference.end())
            rejects<std::invalid_argument>([&] { index.add(id, next.kind, next.a, next.b); });
          else {
            index.add(id, next.kind, next.a, next.b);
            next.order = order++;
            reference.emplace(id, next);
          }
        } else if (operation < 5) {
          check(index.update(id, next.kind, next.a, next.b) == (found != reference.end()));
          if (found != reference.end()) { next.order = found->second.order; found->second = next; }
        } else if (operation == 5) {
          check(index.erase(id) == (reference.erase(id) != 0));
        } else if (operation == 6) {
          const std::size_t n = random() % (reference.size() + 2);
          const double elapsed = (random() % 20000) / 16.0;
          if (n > reference.size())
            rejects<std::invalid_argument>([&] { index.pop_n(n, elapsed); });
          else {
            std::vector<Id> expected;
            auto rank = [&](const auto& item) {
              const auto& entry = item.second;
              const double value = entry.kind == Kind::Tracked
                  ? std::min(entry.a - elapsed / 1.25, entry.b - elapsed / 3.5)
                  : entry.order;
              return std::tuple{static_cast<int>(entry.kind), value, item.first};
            };
            for (std::size_t i = 0; i < n; ++i) {
              auto best = std::min_element(reference.begin(), reference.end(),
                  [&](const auto& a, const auto& b) { return rank(a) < rank(b); });
              expected.push_back(best->first);
              reference.erase(best);
            }
            check(index.pop_n(n, elapsed) == expected);
          }
        } else { index.clear(); reference.clear(); order = 0; }
        check(index.size() == reference.size());
      }
    });
    std::cout << "PASS total " << groups << " groups, " << checks << " checks\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL after " << groups << " groups: " << error.what() << '\n';
    return 1;
  }
}
