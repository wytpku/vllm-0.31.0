#pragma once

#include<cstddef>
#include<cstdint>
#include<set>
#include<unordered_map>
#include<utility>
#include<vector>

namespace vllm{
    class MWFCandidateIndex{
    public:
        using BlockId = std::int64_t;
        enum class Kind { Unhashed, Missing, Tracked };

        MWFCandidateIndex(double alpha, double beta);
        void add(BlockId id, Kind kind, double key_a = 0.0, double key_b = 0.0); // 块加入真是空闲队列后登记
        bool update(BlockId id, Kind kind, double key_a = 0.0, double key_b = 0.0); // 更新已有候选的分类或评分
        bool erase(BlockId id); // 删除候选索引
        std::vector<BlockId> pop_n(std::size_t n, double elapsed); // 按优先级选出并移除n个候选，返回块编号
        void clear(); // 清空索引
        std::size_t size() const{ return entries_.size(); };

    private:
        struct Entry{
            Kind kind;
            std::uint64_t free_order;
            double key_a;
            double key_b;
        };
        using ScoreKey = std::pair<double, BlockId>;
        using OrderKey = std::pair<std::uint64_t, BlockId>;

        double alpha_, beta_;
        std::uint64_t next_free_order_ = 0;
        std::unordered_map<BlockId, Entry> entries_; // 按块编号找到候选记录
        std::set<ScoreKey> scores_a_, scores_b_;
        std::set<OrderKey> unhashed_, missing_;
    };
}
