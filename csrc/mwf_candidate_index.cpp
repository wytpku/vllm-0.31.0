#include "mwf_candidate_index.h"
#include <cmath>
#include <stdexcept>
#include <limits>
namespace vllm{
MWFCandidateIndex::MWFCandidateIndex(double alpha, double beta): alpha_(alpha), beta_(beta){
    if(!std::isfinite(alpha_) || !std::isfinite(beta_) || alpha_ <= 0.0 || beta_ <= 0.0){
        throw std::invalid_argument("MWF alpha and beta must be finite and positive");
    }
}

bool MWFCandidateIndex::erase(BlockId id){
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;

    const Entry& entry = it->second;
    if(entry.kind == Kind::Tracked){
        scores_a_.erase(ScoreKey{entry.key_a, id});
        scores_b_.erase(ScoreKey{entry.key_b, id});
    }
    else{
        auto& ordered = entry.kind == Kind::Unhashed ? unhashed_ : missing_ ;
        ordered.erase(OrderKey{entry.free_order, id});
    }

    entries_.erase(it);
    return true;
}

void MWFCandidateIndex::clear(){
    scores_a_.clear();
    scores_b_.clear();
    unhashed_.clear();
    missing_.clear();
    entries_.clear();
    next_free_order_ = 0;
}

void MWFCandidateIndex::add(BlockId id, Kind kind, double key_a, double key_b){
    if (id < 0 || (kind != Kind::Unhashed && kind != Kind::Missing && kind != Kind::Tracked)){
        throw std::invalid_argument("Invalid block ID or condidate kind");
    }
    if (kind == Kind::Tracked && (!std::isfinite(key_a) || !std::isfinite(key_b))){
        throw std::invalid_argument("MWF score keys must be finite");
    }
    if (entries_.count(id) != 0) {
        throw std::invalid_argument("Block is already a free candidate");
    }
    if (next_free_order_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("MWF free order counter exhausted");
    }
    const auto order = next_free_order_;
    entries_.emplace(id, Entry{kind, order, key_a, key_b});
    try {
        if (kind == Kind::Tracked) {
            scores_a_.emplace(key_a, id);
            scores_b_.emplace(key_b, id);
        } else {
            auto& ordered = kind == Kind::Unhashed ? unhashed_ : missing_;
            ordered.emplace(order, id);
        }
    } catch (...) {
        erase(id);
        throw;
    }
    ++next_free_order_;
}

bool MWFCandidateIndex::update(BlockId id, Kind kind, double key_a, double key_b){
    auto it = entries_.find(id);
    if(it == entries_.end()) return false;
    if (kind != Kind::Unhashed && kind != Kind::Missing &&
        kind != Kind::Tracked) {
        throw std::invalid_argument("Invalid candidate kind");
    }
    if (kind == Kind::Tracked &&
        (!std::isfinite(key_a) || !std::isfinite(key_b))) {
        throw std::invalid_argument("MWF score keys must be finite");
    }
    const Entry old = it->second;
    const OrderKey order{old.free_order, id};

    // 插入新的
    auto& ordered = kind == Kind::Unhashed ? unhashed_ : missing_;
    bool added_a = false, added_b = false, added_order = false;
    try{
        if(kind == Kind::Tracked){
            added_a = scores_a_.emplace(key_a, id).second; // 表示是否真的插入了新节点。如果评分键没变，返回值就是false
            added_b = scores_b_.emplace(key_b, id).second;
        } else{
            added_order = ordered.insert(order).second;
        }
    } catch (...) {
        if (added_a) scores_a_.erase(ScoreKey{key_a, id});
        if (added_b) scores_b_.erase(ScoreKey{key_b, id});
        if (added_order) ordered.erase(order);
        throw;
    }

    // 删除旧的
    if (old.kind == Kind::Tracked){
        if(kind != Kind::Tracked || old.key_a != key_a)
            scores_a_.erase(ScoreKey{old.key_a, id});
        if(kind != Kind::Tracked || old.key_b != key_b)
            scores_b_.erase(ScoreKey{old.key_b, id});
    }
    else if(old.kind != kind){
        auto &previous = old.kind == Kind::Unhashed ? unhashed_ : missing_;
        previous.erase(order);
    }
    it->second = Entry{kind, old.free_order, key_a, key_b};
    return true;
}   

std::vector<MWFCandidateIndex::BlockId> MWFCandidateIndex::pop_n(std::size_t n, double elapsed){
    if(n > entries_.size())
        throw std::invalid_argument("Not enough free candidates");
    if(n == 0) return {};
    if(!std::isfinite(elapsed) || elapsed < 0.0)
        throw std::invalid_argument("Elapsed time must be finite and nonnegative");

    const double decay_a = elapsed / alpha_, decay_b = elapsed / beta_;
    if (!std::isfinite(decay_a) || !std::isfinite(decay_b) ||
            (!scores_a_.empty() && !std::isfinite(scores_a_.begin()->first - decay_a)) ||
            (!scores_b_.empty() && !std::isfinite(scores_b_.begin()->first - decay_b)))
        throw std::overflow_error("MWF score adjustment overflow");
    
    std::vector<BlockId> result;
    result.reserve(n);
    while(result.size() < n){
        BlockId id;
        if(!unhashed_.empty()){
            id = unhashed_.begin()->second;
        }
        else if(!missing_.empty()){
            id = missing_.begin()->second;
        }
        else{
            if(scores_a_.empty() || scores_b_.empty())
                throw std::logic_error("MWF score indexes are inconsistent");
            const auto a = *scores_a_.begin();
            const auto b = *scores_b_.begin();
            const ScoreKey current_a{a.first - decay_a, a.second};
            const ScoreKey current_b{b.first - decay_b, b.second};
            id = (current_a < current_b ? current_a : current_b).second;
        }
        if(!erase(id))
            throw std::logic_error("Selected block has no candidate record");
        result.push_back(id);
    }
    return result;
}
}
