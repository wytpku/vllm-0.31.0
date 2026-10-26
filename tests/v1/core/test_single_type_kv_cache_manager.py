# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import unittest
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.mwf_eviction import MWFEvictionManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CircularBufferManager,
    FullAttentionManager,
    MambaManager,
    MultiSegHitSnapshot,
    RSWAManager,
    SlidingWindowManager,
    plan_multiseg_allocation,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    RSWASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def test_external_computed_blocks_do_not_corrupt_free_pool():
    block_size = 4
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10,
        enable_caching=False,
        hash_block_size=block_size,
    )
    manager = FullAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    request_id = "request"
    manager.allocate_new_blocks(
        request_id,
        num_tokens=3 * block_size,
        num_tokens_main_model=3 * block_size,
    )
    num_free_blocks = block_pool.get_num_free_blocks()

    # Speculative allocations can exceed the blocks implied by the eventual
    # external computed-token count. This must not request a negative number
    # of blocks, which inflates the free-queue counter.
    manager.allocate_external_computed_blocks(
        request_id,
        num_local_computed_tokens=0,
        num_external_computed_tokens=block_size,
    )

    assert block_pool.get_num_free_blocks() == num_free_blocks
    assert len(manager.req_to_blocks[request_id]) == 3


def test_mamba_speculative_block_relocation_requires_exclusive_ownership():
    spec = MambaSpec(
        block_size=4,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=1,
    )
    block_pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=4)
    manager = MambaManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=4,
    )
    block = block_pool.get_new_blocks(1)[0]
    blocks = [block]

    manager._relocate_speculative_block(blocks, 0)

    assert blocks == [block_pool.null_block, block]
    assert block.ref_cnt == 1

    pinned_block = block_pool.get_new_blocks(1)[0]
    block_pool.touch((pinned_block,))
    with pytest.raises(AssertionError, match="exclusively owned and unhashed"):
        manager._relocate_speculative_block([pinned_block], 0)


def test_mamba_retirement_crosses_null_gaps():
    spec = MambaSpec(
        block_size=4,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    pool = BlockPool(num_gpu_blocks=8, enable_caching=False, hash_block_size=4)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=4,
    )
    old, committed, in_flight = pool.get_new_blocks(3)
    manager.req_to_blocks["r"] = [old, pool.null_block, committed, in_flight]

    # The state at token 12 and the following in-flight state must survive.
    for _ in range(2):
        manager.remove_skipped_blocks("r", processed_computed_tokens=12)
        assert manager.req_to_blocks["r"] == [
            pool.null_block,
            pool.null_block,
            committed,
            in_flight,
        ]
        assert old.ref_cnt == 0
        assert committed.ref_cnt == in_flight.ref_cnt == 1
        assert pool.get_num_free_blocks() == 5

    manager.free("r")
    manager.req_to_blocks["r"] = pool.get_new_blocks(2)
    manager.remove_skipped_blocks("r", processed_computed_tokens=5)
    assert manager.req_to_blocks["r"][0].is_null
    assert manager.req_to_blocks["r"][1].ref_cnt == 1


@pytest.mark.parametrize("block_size", [3584, 4608])
@pytest.mark.parametrize("in_flight_chunks", [0, 1, 2])
def test_mamba_retirement_bounds_prefill_states(block_size, in_flight_chunks):
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=5,
    )
    pool = BlockPool(num_gpu_blocks=1000, enable_caching=True, hash_block_size=256)
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    initial_free = pool.get_num_free_blocks()
    chunk = 8192 // block_size * block_size
    peak_held = 0
    # Replay full prefill chunks with an async committed-token lag.
    for target in range(chunk, 480031 + 1, chunk):
        computed = target - chunk
        committed = max(0, computed - in_flight_chunks * chunk)
        manager.remove_skipped_blocks("r", committed)
        manager.get_num_blocks_to_allocate(
            "r", target + 5, [], computed, computed, target
        )
        manager.allocate_new_blocks("r", target + 5, target)
        peak_held = max(peak_held, initial_free - pool.get_num_free_blocks())

    # Five speculative blocks, current/previous states, and bounded in-flight states.
    assert peak_held == 7 + in_flight_chunks
    manager.free("r")
    assert pool.get_num_free_blocks() == initial_free


@pytest.mark.parametrize("block_size", [896, 1536])
@pytest.mark.parametrize("num_speculative_blocks", [0, 1, 4])
@pytest.mark.parametrize("prompt_tokens", [25121, 704547])
def test_mamba_checkpoint_admission_matches_allocation(
    block_size, num_speculative_blocks, prompt_tokens
):
    """Checkpoint admission must match the subsequent physical allocation."""
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=64,
    )
    pool = BlockPool(
        num_gpu_blocks=2048,
        enable_caching=True,
        hash_block_size=128,
    )
    manager = MambaManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    request_id = "prefill"
    computed_tokens = 23040

    def estimate(num_tokens, total_computed_tokens, apply_admission_cap):
        return manager.get_num_blocks_to_allocate(
            request_id=request_id,
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=total_computed_tokens,
            num_local_computed_tokens=total_computed_tokens,
            num_tokens_main_model=num_tokens,
            apply_admission_cap=apply_admission_cap,
        )

    estimate(computed_tokens, 0, False)
    manager.allocate_new_blocks(request_id, computed_tokens, computed_tokens)
    assert request_id in manager._allocated_block_reqs

    admission_estimate = estimate(prompt_tokens, computed_tokens, True)
    allocation_estimate = estimate(prompt_tokens, computed_tokens, False)
    assert request_id in manager._checkpoints

    free_before = pool.get_num_free_blocks()
    manager.allocate_new_blocks(request_id, prompt_tokens, prompt_tokens)
    allocated = free_before - pool.get_num_free_blocks()

    assert admission_estimate == allocation_estimate == allocated


def get_sliding_window_manager(
    sliding_window_spec,
    block_pool,
    enable_caching=True,
    needs_kv_cache_zeroing=False,
):
    # Tests don't exercise admission gating; pass a large cap that is a no-op.
    return SlidingWindowManager(
        sliding_window_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=sliding_window_spec.block_size,
        needs_kv_cache_zeroing=needs_kv_cache_zeroing,
        max_admission_blocks_per_request=10**9,
    )


def get_chunked_local_attention_manager(
    chunked_local_attention_spec,
    block_pool,
    enable_caching=True,
    needs_kv_cache_zeroing=False,
):
    return ChunkedLocalAttentionManager(
        chunked_local_attention_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=chunked_local_attention_spec.block_size,
        needs_kv_cache_zeroing=needs_kv_cache_zeroing,
        max_admission_blocks_per_request=10**9,
    )


def test_circular_buffer_allocates_one_block_for_the_request_lifetime():
    block_size = 4
    spec = CircularBufferSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10, enable_caching=True, hash_block_size=block_size
    )
    manager = CircularBufferManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
        needs_kv_cache_zeroing=True,
        max_admission_blocks_per_request=1,
    )
    request_id = "request"

    assert (
        manager.get_num_blocks_to_allocate(
            request_id,
            num_tokens=128,
            new_computed_blocks=[block_pool.blocks[1]],
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=128,
        )
        == 1
    )
    blocks = manager.allocate_new_blocks(
        request_id, num_tokens=128, num_tokens_main_model=128
    )
    assert len(blocks) == 1
    assert not manager.records_new_block_ids
    assert manager.take_new_block_ids() == []

    for num_tokens in (256, 1024):
        assert (
            manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                [],
                total_computed_tokens=num_tokens - 1,
                num_local_computed_tokens=0,
                num_tokens_main_model=num_tokens,
            )
            == 0
        )
        assert manager.allocate_new_blocks(request_id, num_tokens, num_tokens) == []

    manager.cache_blocks(request_id, 1024, replay_boundaries=(0,))
    manager.remove_skipped_blocks(request_id, 1024)
    assert manager.get_num_common_prefix_blocks(request_id) == 0
    assert manager.get_num_skipped_tokens(1024) == 0
    assert manager.req_to_blocks[request_id] == blocks


def test_sliding_window_records_new_blocks_for_zeroing():
    block_size = 2
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10, enable_caching=False, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(
        spec,
        block_pool,
        enable_caching=False,
        needs_kv_cache_zeroing=True,
    )

    blocks = manager.allocate_new_blocks(
        "request", num_tokens=4, num_tokens_main_model=4
    )

    assert manager.records_new_block_ids
    assert manager.take_new_block_ids() == [block.block_id for block in blocks]
    assert manager.take_new_block_ids() == []


def test_chunked_local_attention_records_new_blocks_for_zeroing():
    block_size = 2
    spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )
    block_pool = BlockPool(
        num_gpu_blocks=10, enable_caching=False, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(
        spec,
        block_pool,
        enable_caching=False,
        needs_kv_cache_zeroing=True,
    )

    blocks = manager.allocate_new_blocks(
        "request", num_tokens=4, num_tokens_main_model=4
    )

    assert manager.records_new_block_ids
    assert manager.take_new_block_ids() == [block.block_id for block in blocks]
    assert manager.take_new_block_ids() == []


def test_chunked_local_attention_possible_cached_prefix():
    block_size = 2
    chunked_local_attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(
        chunked_local_attention_spec, block_pool
    )

    def run_one_case(block_is_cached, tail_token, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size + tail_token,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=chunked_local_attention_spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )[0][0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: (expect_length - 1) // 2]
        )

    run_one_case([True], 0, 1)
    run_one_case([True], 1, 1)
    run_one_case([True, False], 0, 2)
    run_one_case([True, False], 1, 2)
    run_one_case([True, True], 0, 2)
    run_one_case([True, True], 1, 2)
    run_one_case([True, True, False], 0, 2)
    run_one_case([True, True, False], 1, 2)
    run_one_case([True, True, True], 0, 3)
    run_one_case([True, True, True], 1, 3)
    run_one_case([True, True, True, False], 0, 4)
    run_one_case([True, True, True, False], 1, 4)
    run_one_case([random.choice([True, False])] * 8 + [True], 1, 9)
    run_one_case([random.choice([True, False])] * 8 + [False], 1, 8)
    run_one_case([random.choice([True, False])] * 8 + [True, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 1, 10)


def test_sliding_window_possible_cached_prefix():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    def run_one_case(block_is_cached, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=sliding_window_spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )[0][0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: expect_length - 2]
        )
        for i in range(2):
            if i < expect_length:
                block_index = expect_length - i - 1
                assert computed_blocks[block_index].block_id == block_index + 10

    run_one_case([False] * 10, 0)
    run_one_case([True], 1)
    run_one_case([True, False], 1)
    run_one_case([True, True], 2)
    run_one_case([True, True, False], 2)
    run_one_case([True, True, True], 3)
    run_one_case([True, True, True, False], 3)
    run_one_case(
        [True, True, False, True, False, False, True, True, False, True, True, True], 12
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False], 8
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False, True],
        8,
    )


def test_sliding_window_cache_hit_with_finer_hash_alignment():
    """Sliding-window lookup uses full blocks with finer hybrid-cache hashes."""
    hash_block_size = 2
    block_size = 4
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=8,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)
    block_hashes = [BlockHash(str(i).encode()) for i in range(4)]

    # Each physical block uses the last chained hash in its block-size view.
    for block_hash, block in zip(
        (block_hashes[1], block_hashes[3]), block_pool.blocks[10:12]
    ):
        block_pool.cached_block_hash_to_block.insert(
            make_block_hash_with_group_id(block_hash, 0), block
        )

    computed_blocks, hit_length = manager.find_longest_cache_hit(
        block_hashes=block_hashes,
        max_length=8,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=sliding_window_spec,
        drop_eagle_block=False,
        alignment_tokens=hash_block_size,
    )

    assert hit_length == 8
    assert computed_blocks[0] == block_pool.blocks[10:12]


def test_chunked_local_attention_remove_skipped_blocks():
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_chunked_local_attention_manager(attention_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # For 4th token (0-indexed), token 0-3 is out of the local attention window.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, [null_block_id] * 2)

    # For 6th token (0-indexed), token 4 - 6 are in local attention window,
    # token 0 - 3 are out, 2 blocks can be removed.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])
    # For 12th token (0-indexed),
    # token 0-11 are out, 6 block can be removed.
    manager.remove_skipped_blocks("test", 12)
    assert_block_id(block_table, [null_block_id] * 6)


def test_sliding_window_remove_skipped_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # 4 tokens are computed. Only token 0 is out of the sliding window. As
    # block 1000 also contains token 1 that is in the sliding window, block 1000
    # cannot be removed.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, original_block_ids)

    # 5 tokens are computed. Token 0 & 1 are out of the sliding window.
    # Block 1000 can be removed.
    manager.remove_skipped_blocks("test", 5)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 6 tokens are computed. Token 0-2 are out of the sliding window.
    # Cannot remove new block as the block 1001 is still used by token 3.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 7 tokens are computed. Token 0-3 are out of the sliding window.
    # Block 1001 can be removed and block 1000 is already removed.
    manager.remove_skipped_blocks("test", 7)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])

    # 11 tokens are computed. Token 0-7 are out of the sliding window.
    # Block 1002 & 1003 can be removed now. Block 1003 represents a longer
    # sequence, and is expected to be evicted earlier than 1002, so the order
    # of removed blocks should be [1003, 1002].
    manager.remove_skipped_blocks("test", 11)
    assert_block_id(block_table, [null_block_id] * 4 + original_block_ids[4:])


def test_rswa_remove_skipped_blocks_gap_range():
    block_size = 4
    rswa_spec = RSWASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        rswa_window=8,
    )
    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=4)
    manager = RSWAManager(
        rswa_spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )

    null_block_id = block_pool.null_block.block_id
    original_block_ids = list(range(1000, 1010))
    block_table = [
        KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
        for id_ in original_block_ids
    ]
    manager.req_to_blocks["test"] = block_table

    prefix_len = 16

    # Without num_prompt_tokens, R-SWA does not evict gap blocks.
    manager.remove_skipped_blocks("test", 28)
    assert [b.block_id for b in block_table] == original_block_ids

    # Gap = block 4 only (tokens [16, 20) fall in the gap).
    manager.remove_skipped_blocks("test", 28, num_prompt_tokens=prefix_len)
    expected = original_block_ids.copy()
    expected[4] = null_block_id
    assert [b.block_id for b in block_table] == expected

    # Window moves: blocks 5 and 6 also enter the gap; block 4 is already null.
    manager.remove_skipped_blocks("test", 36, num_prompt_tokens=prefix_len)
    expected[5] = null_block_id
    expected[6] = null_block_id
    assert [b.block_id for b in block_table] == expected


def test_get_num_blocks_to_allocate():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 0, 20 * block_size
        )
        == 15
    )


def test_evictable_cached_blocks_not_double_allocated():
    block_size = 2
    sliding_window_length = 2 * block_size
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window_length,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    request_id = "req"
    evictable_block = block_pool.blocks[1]  # ref_cnt == 0, eviction candidate

    num_blocks_to_allocate = manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=2 * block_size,
        new_computed_blocks=[evictable_block],
        total_computed_tokens=block_size,
        num_local_computed_tokens=block_size,
        num_tokens_main_model=2 * block_size,
    )
    # Free capacity check should count evictable cached blocks, but allocation
    # should only allocate the truly new block.
    assert num_blocks_to_allocate == 2

    manager.add_local_computed_blocks(
        request_id,
        [evictable_block],
        num_local_computed_tokens=block_size,
        num_external_computed_tokens=0,
    )
    new_blocks = manager.allocate_new_blocks(
        request_id, num_tokens=4, num_tokens_main_model=4
    )
    assert len(new_blocks) == 1
    assert len(manager.req_to_blocks[request_id]) == 2


class TestMultiSegAllocationPlan(unittest.TestCase):
    def hit(self, index, block_id, is_free=True):
        key = make_block_hash_with_group_id(BlockHash(b"expected"), 0)
        return MultiSegHitSnapshot(index, block_id, key, is_free)

    def test_no_hits_allocate_every_page_including_partial_tail(self):
        for block_size in (8, 16, 32):
            for num_tokens, pages in ((0, 0), (block_size, 1), (block_size + 1, 2)):
                with self.subTest(block_size=block_size, num_tokens=num_tokens):
                    plan = plan_multiseg_allocation(num_tokens, block_size, (), ())
                    self.assertEqual(plan.block_ids, (None,) * pages)
                    self.assertEqual(plan.holes, tuple(range(pages)))
                    self.assertEqual(plan.acquire_block_ids, ())
                    self.assertEqual(plan.free_need, pages)

    def test_nine_hits_leave_one_new_page_with_distinct_free_capacity(self):
        for is_free, free_need in ((True, 10), (False, 1)):
            with self.subTest(is_free=is_free):
                hits = tuple(self.hit(i, i + 1, is_free) for i in range(9))
                plan = plan_multiseg_allocation(160, 16, (), hits)
                self.assertEqual(plan.block_ids, (*range(1, 10), None))
                self.assertEqual(plan.holes, (9,))
                self.assertEqual(plan.acquire_block_ids, tuple(range(1, 10)))
                self.assertEqual(plan.free_need, free_need)

    def test_repeated_physical_hit_acquires_twice_but_uses_free_capacity_once(self):
        hits = (self.hit(1, 7), self.hit(3, 7))
        plan = plan_multiseg_allocation(64, 16, (), hits)
        self.assertEqual(plan.block_ids, (None, 7, None, 7))
        self.assertEqual(plan.holes, (0, 2))
        self.assertEqual(plan.acquire_block_ids, (7, 7))
        self.assertEqual(plan.free_need, 3)

    def test_owned_physical_page_can_acquire_another_logical_reference(self):
        plan = plan_multiseg_allocation(64, 16, (7,), (self.hit(2, 7, False),))
        self.assertEqual(plan.block_ids, (7, None, 7, None))
        self.assertEqual(plan.holes, (1, 3))
        self.assertEqual(plan.acquire_block_ids, (7,))
        self.assertEqual(plan.free_need, 2)

    def test_invalid_sizes_and_owned_prefix_are_rejected(self):
        for num_tokens, block_size, owned in (
            (-1, 16, ()),
            (16, 0, ()),
            (16, -1, ()),
            (16, 16, (1, 2)),
            (16, 16, (-1,)),
        ):
            with self.subTest(
                num_tokens=num_tokens, block_size=block_size, owned=owned
            ):
                with self.assertRaises(ValueError):
                    plan_multiseg_allocation(num_tokens, block_size, owned, ())

    def test_invalid_hit_layouts_are_rejected(self):
        cases = (
            (64, (), (self.hit(-1, 7),)),
            (64, (), (self.hit(4, 7),)),
            (17, (), (self.hit(1, 7),)),
            (64, (), (self.hit(1, -1),)),
            (64, (1,), (self.hit(0, 7),)),
            (64, (), (self.hit(1, 7), self.hit(1, 8))),
            (64, (), (self.hit(1, 7), self.hit(2, 7, False))),
            (64, (7,), (self.hit(2, 7),)),
        )
        for num_tokens, owned, hits in cases:
            with self.subTest(num_tokens=num_tokens, owned=owned, hits=hits):
                with self.assertRaises(ValueError):
                    plan_multiseg_allocation(num_tokens, 16, owned, hits)

    def test_plan_retains_identity_and_ownership_snapshots(self):
        owned = [1]
        hit = self.hit(2, 7)
        hits = [hit]
        plan = plan_multiseg_allocation(49, 16, owned, hits)
        owned[0] = 99
        hits.clear()
        self.assertEqual(plan.block_ids, (1, None, 7, None))
        self.assertEqual(plan.hits, (hit,))
        self.assertEqual(
            plan.hits[0].expected_hash_with_group, hit.expected_hash_with_group
        )
        with self.assertRaises(FrozenInstanceError):
            setattr(hit, "block_id", 99)
        with self.assertRaises(FrozenInstanceError):
            setattr(plan, "block_ids", ())


class TestMultiSegHitRefresh(unittest.TestCase):
    def fixture(self, block_size=16, hash_block_size=16, **spec_options):
        pool = BlockPool(4, enable_caching=True, hash_block_size=hash_block_size)
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            **spec_options,
        )
        manager = FullAttentionManager(spec, pool, True, 0, block_size)
        request = SimpleNamespace(
            request_id="msa-request",
            num_tokens=65,
            block_hashes=[BlockHash(bytes([i])) for i in range(4)],
        )
        key = make_block_hash_with_group_id(
            request.block_hashes[block_size // hash_block_size - 1], 0
        )
        block = pool.blocks[1]
        pool._insert_block_hash(key, block, num_tokens=block_size)
        hit = MultiSegHitSnapshot(0, block.block_id, key, True)
        return manager, request, block, hit

    def test_refresh_reads_current_refs_without_acquiring_ownership(self):
        manager, request, block, hit = self.fixture()
        pool = manager.block_pool
        for ref_count in (0, 1, 0):
            if ref_count:
                pool.touch([block])
            elif block.ref_cnt:
                pool.free_blocks([block])
            free_before = tuple(
                b.block_id for b in pool.free_block_queue.get_all_free_blocks()
            )
            refreshed = manager._refresh_multiseg_hit(request, hit)
            self.assertEqual(refreshed, replace(hit, is_free=ref_count == 0))
            self.assertEqual(block.ref_cnt, ref_count)
            self.assertEqual(
                tuple(b.block_id for b in pool.free_block_queue.get_all_free_blocks()),
                free_before,
            )
            self.assertTrue(
                pool.cached_block_hash_to_block.contain(
                    hit.expected_hash_with_group, hit.block_id
                )
            )
        self.assertTrue(hit.is_free)
        self.assertEqual(dict(manager.req_to_blocks), {})
        self.assertEqual(manager.new_block_ids, [])

    def test_changed_request_or_saved_hash_group_cannot_rebind_candidate(self):
        manager, request, block, hit = self.fixture()
        for bad_key in (
            make_block_hash_with_group_id(BlockHash(b"changed"), 0),
            make_block_hash_with_group_id(request.block_hashes[0], 1),
        ):
            with self.subTest(bad_key=bad_key):
                bad_hit = replace(hit, expected_hash_with_group=bad_key)
                self.assertIsNone(manager._refresh_multiseg_hit(request, bad_hit))
        request.block_hashes[0] = BlockHash(b"changed")
        manager.block_pool._insert_block_hash(
            make_block_hash_with_group_id(request.block_hashes[0], 0), block, 16
        )
        self.assertIsNone(manager._refresh_multiseg_hit(request, hit))

    def test_removed_mapping_or_other_physical_block_does_not_match(self):
        manager, request, block, hit = self.fixture()
        cache = manager.block_pool.cached_block_hash_to_block
        cache.pop(hit.expected_hash_with_group, block.block_id)
        self.assertIsNone(manager._refresh_multiseg_hit(request, hit))
        other = manager.block_pool.blocks[2]
        other.set_block_hash(hit.expected_hash_with_group)
        cache.insert(hit.expected_hash_with_group, other)
        self.assertIsNone(manager._refresh_multiseg_hit(request, hit))
        cache.insert(hit.expected_hash_with_group, block)
        self.assertEqual(manager._refresh_multiseg_hit(request, hit), hit)

    def test_alias_identity_does_not_require_matching_primary_hash(self):
        manager, request, block, hit = self.fixture()
        pool = manager.block_pool
        pool.cached_block_hash_to_block.pop(
            hit.expected_hash_with_group, block.block_id
        )
        block.reset_hash()
        primary = make_block_hash_with_group_id(BlockHash(b"primary"), 0)
        pool._insert_block_hash(primary, block, 16)
        pool._insert_block_hash(hit.expected_hash_with_group, block, 16)
        self.assertNotEqual(block.block_hash, hit.expected_hash_with_group)
        self.assertEqual(manager._refresh_multiseg_hit(request, hit), hit)

    def test_hashes_are_resolved_at_page_granularity_for_raw_and_view_inputs(self):
        manager, request, block, hit = self.fixture(block_size=32)
        self.assertEqual(manager._refresh_multiseg_hit(request, hit), hit)
        request.block_hashes = BlockHashListWithBlockSize(request.block_hashes, 16, 32)
        self.assertEqual(manager._refresh_multiseg_hit(request, hit), hit)

    def test_invalid_ids_and_unusable_canonical_blocks_are_rejected(self):
        for change in (
            "negative_id", "large_id", "null", "negative_ref", "foreign_pool"
        ):
            with self.subTest(change=change):
                manager, request, block, hit = self.fixture()
                if change in ("negative_id", "large_id"):
                    hit = replace(hit, block_id=-1 if change == "negative_id" else 99)
                elif change == "null":
                    block.is_null = True
                elif change == "negative_ref":
                    block.ref_cnt = -1
                else:
                    block.pool = object()
                self.assertIsNone(manager._refresh_multiseg_hit(request, hit))

    def test_logits_page_and_incomplete_tail_are_not_reused(self):
        for num_tokens in (0, 1, 15, 16):
            with self.subTest(num_tokens=num_tokens):
                manager, request, block, hit = self.fixture()
                request.num_tokens = num_tokens
                self.assertIsNone(manager._refresh_multiseg_hit(request, hit))

    def test_unsupported_attention_modes_are_rejected(self):
        for options in (
            {"sliding_window": 16},
            {"attention_chunk_size": 16},
            {"non_causal": True},
            {"dcp_world_size": 2},
            {"pcp_world_size": 2},
        ):
            with self.subTest(options=options):
                spec_options = {
                    k: v for k, v in options.items() if not k.endswith("world_size")
                }
                manager, request, block, hit = self.fixture(**spec_options)
                for name in ("dcp_world_size", "pcp_world_size"):
                    if name in options:
                        setattr(manager, name, options[name])
                with self.assertRaises(NotImplementedError):
                    manager._refresh_multiseg_hit(request, hit)


class TestMultiSegPlanningAdapter(unittest.TestCase):
    fixture = TestMultiSegHitRefresh.fixture

    def state(self, manager):
        pool = manager.block_pool
        return (
            tuple((b.ref_cnt, b.block_hash) for b in pool.blocks),
            tuple(b.block_id for b in pool.free_block_queue.get_all_free_blocks()),
            {
                key: tuple(b.block_id for b in blocks)
                for key, blocks in manager.req_to_blocks.items()
            },
            {
                key: tuple(sorted(value)) if isinstance(value, dict)
                else (value.block_id,)
                for key, value in pool.cached_block_hash_to_block._cache.items()
            },
            tuple(manager.new_block_ids),
            dict(manager.num_cached_block),
        )

    def test_empty_plan_does_not_create_request_or_change_pool(self):
        manager, request, block, hit = self.fixture()
        before = self.state(manager)
        plan = manager.plan_multiseg_blocks(request, 17, ())
        self.assertEqual(plan.block_ids, (None, None))
        self.assertEqual(plan.free_need, 2)
        self.assertEqual(self.state(manager), before)

    def test_stale_candidate_becomes_hole_and_live_refs_refresh_capacity(self):
        for shared, free_need in ((False, 2), (True, 1)):
            with self.subTest(shared=shared):
                manager, request, block, hit = self.fixture()
                pool = manager.block_pool
                if shared:
                    pool.touch([block])
                before = self.state(manager)
                plan = manager.plan_multiseg_blocks(request, 32, (hit,))
                self.assertEqual(plan.block_ids, (block.block_id, None))
                self.assertEqual(plan.hits, (replace(hit, is_free=not shared),))
                self.assertEqual(plan.free_need, free_need)
                self.assertEqual(self.state(manager), before)
                pool.evict_blocks({block.block_id})
                before = self.state(manager)
                plan = manager.plan_multiseg_blocks(request, 32, (hit,))
                self.assertEqual(plan.block_ids, (None, None))
                self.assertEqual(plan.hits, ())
                self.assertEqual(plan.free_need, 2)
                self.assertEqual(self.state(manager), before)

    def test_owned_page_reuse_acquires_only_the_new_logical_reference(self):
        manager, request, block, hit = self.fixture()
        pool = manager.block_pool
        pool.touch([block])
        manager.req_to_blocks[request.request_id] = [block]
        key = make_block_hash_with_group_id(request.block_hashes[2], 0)
        pool._insert_block_hash(key, block, 48)
        hit = MultiSegHitSnapshot(2, block.block_id, key, True)
        before = self.state(manager)
        plan = manager.plan_multiseg_blocks(request, 64, (hit,))
        self.assertEqual(plan.block_ids, (block.block_id, None, block.block_id, None))
        self.assertEqual(plan.acquire_block_ids, (block.block_id,))
        self.assertEqual(plan.free_need, 2)
        self.assertEqual(self.state(manager), before)

    def test_partial_and_future_candidates_remain_available_for_later_window(self):
        manager, request, block, hit = self.fixture()
        pool = manager.block_pool
        later = pool.blocks[2]
        key = make_block_hash_with_group_id(request.block_hashes[2], 0)
        pool._insert_block_hash(key, later, 48)
        hits = [hit, MultiSegHitSnapshot(2, later.block_id, key, True)]
        before = self.state(manager)
        plan = manager.plan_multiseg_blocks(request, 33, hits)
        self.assertEqual(plan.block_ids, (block.block_id, None, None))
        plan = manager.plan_multiseg_blocks(request, 49, hits)
        self.assertEqual(plan.block_ids, (block.block_id, None, later.block_id, None))
        self.assertEqual(plan.free_need, 4)
        self.assertEqual(len(hits), 2)
        self.assertEqual(self.state(manager), before)

    def test_bad_future_candidates_are_rejected_before_window_filtering(self):
        manager, request, block, hit = self.fixture()
        future = replace(hit, logical_index=3)
        for hits in (
            (future, future),
            (replace(future, logical_index=-1),),
            (replace(future, logical_index=5),),
            (replace(future, block_id=-1),),
            (replace(future, block_id=99),),
        ):
            with self.subTest(hits=hits):
                before = self.state(manager)
                with self.assertRaises(ValueError):
                    manager.plan_multiseg_blocks(request, 16, hits)
                self.assertEqual(self.state(manager), before)

    def test_invalid_owned_pages_and_owned_logical_overlap_are_rejected(self):
        for kind in ("zero_ref", "null", "foreign", "noncanonical", "overlap"):
            with self.subTest(kind=kind):
                manager, request, block, hit = self.fixture()
                if kind != "zero_ref":
                    manager.block_pool.touch([block])
                owned = block
                if kind == "null":
                    block.is_null = True
                elif kind == "foreign":
                    block.pool = object()
                elif kind == "noncanonical":
                    owned = KVCacheBlock(block.block_id, ref_cnt=1, pool=block.pool)
                manager.req_to_blocks[request.request_id] = [owned]
                before = self.state(manager)
                with self.assertRaises(ValueError):
                    manager.plan_multiseg_blocks(
                        request, 32, (hit,) if kind == "overlap" else ()
                    )
                self.assertEqual(self.state(manager), before)

    def test_invalid_window_and_pending_cow_are_rejected(self):
        manager, request, block, hit = self.fixture()
        for num_tokens in (-1, request.num_tokens + 1):
            with self.subTest(num_tokens=num_tokens), self.assertRaises(ValueError):
                manager.plan_multiseg_blocks(request, num_tokens, ())
        other = manager.block_pool.blocks[2]
        manager.block_pool.touch([block, other])
        manager.req_to_blocks[request.request_id] = [block, other]
        with self.assertRaises(ValueError):
            manager.plan_multiseg_blocks(request, 16, ())
        manager.req_to_blocks.clear()
        manager._partial_hit_reqs[request.request_id] = (0, block)
        with self.assertRaises(NotImplementedError):
            manager.plan_multiseg_blocks(request, 32, ())

    def test_empty_hits_still_reject_unsupported_configuration(self):
        for name, value in (
            ("enable_caching", False), ("pool_caching", False),
            ("sliding_window", 16), ("attention_chunk_size", 16),
            ("non_causal", True), ("dcp_world_size", 2), ("pcp_world_size", 2),
        ):
            with self.subTest(name=name):
                manager, request, block, hit = self.fixture()
                if name == "pool_caching":
                    manager.block_pool.enable_caching = value
                elif name in ("sliding_window", "attention_chunk_size", "non_causal"):
                    manager.kv_cache_spec = replace(
                        manager.kv_cache_spec, **{name: value}
                    )
                else:
                    setattr(manager, name, value)
                with self.assertRaises(NotImplementedError):
                    manager.plan_multiseg_blocks(request, 32, ())


class TestMultiSegInstallation(unittest.TestCase):
    def fixture(self, count=9, pool_size=11):
        pool = BlockPool(pool_size, enable_caching=True, hash_block_size=16)
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        manager = FullAttentionManager(spec, pool, True, 0, 16)
        manager._record_new_block_ids = True
        request = SimpleNamespace(
            request_id="install", num_tokens=161, num_prompt_tokens=161,
            block_hashes=[BlockHash(bytes([i])) for i in range(10)],
            num_computed_tokens=0, num_in_flight_tokens=0,
            skip_reading_prefix_cache=False, kv_cache_report_mode="incremental",
            status=RequestStatus.WAITING, shared_prefix_boundary=0,
            continuum_suffix_computed_block_locs=[0],
            continuum_multiseg_cached_blocks=None,
        )
        for index in range(1, count + 1):
            key = make_block_hash_with_group_id(request.block_hashes[index], 0)
            pool._insert_block_hash(key, pool.blocks[index], (index + 1) * 16)
        coord = object.__new__(UnitaryKVCacheCoordinator)
        coord.single_type_managers = (manager,)
        coord.block_pool = pool
        coord.kv_cache_spec = spec
        coord.dcp_world_size = coord.pcp_world_size = 1
        coord.block_size = coord.scheduler_block_size = 16
        coord.eagle_group_ids = set()
        coord.num_reprefillable_tokens = 0
        coord.retention_interval = None
        cache = object.__new__(KVCacheManager)
        cache.coordinator = coord
        cache.block_pool = pool
        cache.enable_multiseg_cache = cache.enable_caching = True
        cache.enable_kv_cache_events = cache.log_stats = False
        cache.max_model_len = 1024
        cache.watermark_blocks = 0
        cache.retained_hit_group_ids = ()
        cache.empty_kv_cache_blocks = KVCacheBlocks(((),))
        cache.num_kv_cache_groups = 1
        return cache, manager, request

    def snapshot(self, cache, manager, request):
        pool = cache.block_pool
        return (
            tuple((b.ref_cnt, b.block_hash, b.block_hash_num_tokens)
                  for b in pool.blocks),
            tuple(b.block_id for b in pool.free_block_queue.get_all_free_blocks()),
            {key: tuple(sorted(value)) if isinstance(value, dict)
             else (value.block_id,)
             for key, value in pool.cached_block_hash_to_block._cache.items()},
            {key: frozenset(value) for key, value
             in pool.cached_block_hashes_by_block.items()},
            {key: tuple(b.block_id for b in value)
             for key, value in manager.req_to_blocks.items()},
            dict(manager.num_cached_block),
            tuple(manager.new_block_ids), frozenset(manager._multiseg_requests),
            tuple(request.continuum_suffix_computed_block_locs),
            request.continuum_multiseg_cached_blocks,
            tuple(pool.kv_event_queue),
        )

    def test_sparse_manager_chain_allocates_only_the_hole(self):
        cache, manager, request = self.fixture()
        prefix, count, _ = cache.get_computed_blocks(request)
        self.assertEqual(count, 0)
        original = cache.block_pool.get_new_blocks
        with patch.object(
            cache.block_pool, "get_new_blocks", wraps=original
        ) as allocate:
            result = cache.allocate_slots(
                request, 160, num_new_computed_tokens=count,
                new_computed_blocks=prefix,
            )
        allocate.assert_called_once_with(1)
        self.assertEqual(result.get_block_ids(), ([10, *range(1, 10)],))
        self.assertEqual(manager.new_block_ids, [10])
        self.assertEqual(request.continuum_suffix_computed_block_locs, [16, 160, 0])
        self.assertTrue(all(
            block.ref_cnt == 1 for block in manager.req_to_blocks["install"]
        ))

    def test_duplicate_hit_has_one_capacity_cost_and_two_final_references(self):
        cache, manager, request = self.fixture()
        prefix, count, _ = cache.get_computed_blocks(request)
        key = make_block_hash_with_group_id(request.block_hashes[3], 0)
        cache.block_pool._insert_block_hash(key, cache.block_pool.blocks[2], 64)
        request.continuum_multiseg_cached_blocks = tuple(
            replace(hit, block_id=2) if hit.logical_index == 3 else hit
            for hit in request.continuum_multiseg_cached_blocks
        )
        cache.allocate_slots(request, 160, new_computed_blocks=prefix)
        self.assertEqual(cache.block_pool.blocks[2].ref_cnt, 2)
        self.assertEqual(len(manager.new_block_ids), 1)
        manager.free(request.request_id)
        self.assertTrue(all(
            block.ref_cnt == 0 for block in cache.block_pool.blocks[1:]
        ))

    def test_no_hits_and_prefix_only_hits_keep_dense_metadata(self):
        cache, manager, request = self.fixture(count=0)
        prefix, count, _ = cache.get_computed_blocks(request)
        self.assertEqual(count, 0)
        cache.allocate_slots(request, 160, new_computed_blocks=prefix)
        self.assertEqual(len(manager.new_block_ids), 10)
        self.assertEqual(request.continuum_suffix_computed_block_locs, [0])
        cache, manager, request = self.fixture(count=1, pool_size=4)
        key = make_block_hash_with_group_id(request.block_hashes[0], 0)
        cache.block_pool._insert_block_hash(key, cache.block_pool.blocks[2], 16)
        prefix, count, _ = cache.get_computed_blocks(request)
        self.assertEqual(count, 32)
        cache.allocate_slots(
            request, 16, num_new_computed_tokens=count, new_computed_blocks=prefix
        )
        self.assertEqual(len(manager.new_block_ids), 1)
        self.assertEqual(request.continuum_suffix_computed_block_locs, [32])

    def test_old_plan_rejects_table_or_identity_changes_but_refreshes_refs(self):
        cache, manager, request = self.fixture(count=1, pool_size=4)
        hits = manager.discover_multiseg_hits(request)
        plan = manager.plan_multiseg_blocks(request, 32, hits)
        cache.block_pool.touch([cache.block_pool.blocks[1]])
        refreshed = manager._refresh_multiseg_plan(request, plan)
        self.assertEqual(refreshed.free_need, 1)
        self.assertEqual(plan.free_need, 2)
        manager.req_to_blocks[request.request_id] = [cache.block_pool.blocks[1]]
        with self.assertRaises(ValueError):
            manager._refresh_multiseg_plan(request, plan)
        manager.req_to_blocks.clear()
        cache.block_pool.evict_blocks({1})
        with self.assertRaises(ValueError):
            manager._refresh_multiseg_plan(request, plan)

    def test_foreign_prefix_object_cannot_be_rebound_by_physical_id(self):
        cache, manager, request = self.fixture(count=1, pool_size=4)
        key = make_block_hash_with_group_id(request.block_hashes[0], 0)
        cache.block_pool._insert_block_hash(key, cache.block_pool.blocks[2], 16)
        prefix, count, _ = cache.get_computed_blocks(request)
        forged = KVCacheBlock(prefix.blocks[0][0].block_id, pool=cache.block_pool)
        invalid = KVCacheBlocks(([forged, *prefix.blocks[0][1:]],))
        before = self.snapshot(cache, manager, request)
        with self.assertRaisesRegex(ValueError, "canonical"):
            cache.allocate_slots(
                request, 16, num_new_computed_tokens=count,
                new_computed_blocks=invalid,
            )
        self.assertEqual(self.snapshot(cache, manager, request), before)

    def test_disabled_feature_retains_dense_allocation_and_decode_tail(self):
        cache, manager, request = self.fixture(pool_size=12)
        cache.enable_multiseg_cache = False
        prefix, count, _ = cache.get_computed_blocks(request)
        self.assertEqual(count, 0)
        result = cache.allocate_slots(request, 160, new_computed_blocks=prefix)
        self.assertEqual(len(result.blocks[0]), 10)
        self.assertEqual(len(manager.new_block_ids), 10)
        self.assertEqual(request.continuum_suffix_computed_block_locs, [0])
        self.assertEqual(manager._multiseg_requests, set())
        request.num_computed_tokens = 160
        tail = cache.allocate_slots(request, 1)
        self.assertEqual(len(tail.blocks[0]), 1)
        request.num_computed_tokens = 161
        request.num_tokens = 162
        self.assertEqual(cache.allocate_slots(request, 1).get_block_ids(), ([],))

    def test_reservation_allocation_and_publication_failures_restore_exact_state(self):
        for phase in ("touch", "get_new_blocks", "metadata", "cache"):
            with self.subTest(phase=phase):
                cache, manager, request = self.fixture()
                prefix, count, _ = cache.get_computed_blocks(request)
                before = self.snapshot(cache, manager, request)
                if phase in ("touch", "get_new_blocks"):
                    owner, name = cache.block_pool, phase
                elif phase == "metadata":
                    owner, name = cache, "_publish_multiseg_metadata"
                else:
                    owner, name = cache.coordinator, "cache_blocks"
                original = getattr(owner, name)

                def fail_after_change(*args, **kwargs):
                    original(*args, **kwargs)
                    if phase == "metadata":
                        cache.block_pool.kv_event_queue.append("injected-event")
                    raise RuntimeError("injected failure")

                with patch.object(owner, name, side_effect=fail_after_change):
                    with self.assertRaisesRegex(RuntimeError, "injected failure"):
                        cache.allocate_slots(request, 160, new_computed_blocks=prefix)
                self.assertEqual(self.snapshot(cache, manager, request), before)

    def test_victim_alias_and_queue_order_survive_metadata_failure(self):
        cache, manager, request = self.fixture(count=1, pool_size=4)
        pool = cache.block_pool
        victim = pool.blocks[2]
        for key in (b"victim", b"alias"):
            pool._insert_block_hash(make_block_hash_with_group_id(BlockHash(key), 0),
                                    victim, 16)
        prefix, _, _ = cache.get_computed_blocks(request)
        before = self.snapshot(cache, manager, request)
        with patch.object(
            cache, "_publish_multiseg_metadata", side_effect=RuntimeError
        ):
            with self.assertRaises(RuntimeError):
                cache.allocate_slots(request, 48, new_computed_blocks=prefix)
        self.assertEqual(self.snapshot(cache, manager, request), before)

    def test_failed_later_window_restores_existing_table_identity_and_metadata(self):
        for phase in ("_publish_multiseg_metadata", "cache_blocks"):
            with self.subTest(phase=phase):
                cache, manager, request = self.fixture()
                prefix, _, _ = cache.get_computed_blocks(request)
                cache.allocate_slots(request, 32, new_computed_blocks=prefix)
                request.num_computed_tokens = 32
                table = manager.req_to_blocks[request.request_id]
                before = self.snapshot(cache, manager, request)
                owner = cache if phase.startswith("_") else cache.coordinator
                original = getattr(owner, phase)

                def fail_after_change(*args, **kwargs):
                    original(*args, **kwargs)
                    raise RuntimeError("later window")

                with patch.object(owner, phase, side_effect=fail_after_change):
                    with self.assertRaisesRegex(RuntimeError, "later window"):
                        cache.allocate_slots(request, 32)
                self.assertIs(manager.req_to_blocks[request.request_id], table)
                self.assertEqual(self.snapshot(cache, manager, request), before)

    def test_capacity_failure_and_external_observers_do_not_mutate_state(self):
        cache, manager, request = self.fixture(count=1, pool_size=3)
        prefix, _, _ = cache.get_computed_blocks(request)
        before = self.snapshot(cache, manager, request)
        self.assertIsNone(cache.allocate_slots(request, 64, new_computed_blocks=prefix))
        self.assertEqual(self.snapshot(cache, manager, request), before)
        for observer in ("metrics", "watcher"):
            with self.subTest(observer=observer):
                if observer == "metrics":
                    cache.block_pool.metrics_collector = object()
                else:
                    cache.block_pool.metrics_collector = None
                    cache.block_pool._reuse_watchers[2] = lambda block: None
                with self.assertRaises(NotImplementedError):
                    cache.allocate_slots(request, 32, new_computed_blocks=prefix)
                self.assertEqual(self.snapshot(cache, manager, request), before)

    def test_delay_cache_delivery_contains_hits_but_only_new_pages_need_zeroing(self):
        cache, manager, request = self.fixture()
        prefix, _, _ = cache.get_computed_blocks(request)
        result = cache.allocate_slots(
            request, 160, new_computed_blocks=prefix, delay_cache_blocks=True
        )
        self.assertEqual(len(result.blocks[0]), 10)
        self.assertEqual(len(manager.new_block_ids), 1)
        self.assertIsNone(result.blocks[0][0].block_hash)

    def test_mwf_python_metadata_is_restored_and_native_resync_is_requested(self):
        cache, manager, request = self.fixture()
        clock = iter(range(10, 1000))
        policy = MWFEvictionManager(
            minexp_a=0, minexp_b=-19.15349,
            minexp_alpha=1915.332, minexp_beta=47.63942,
            clock=lambda: float(next(clock)),
        )
        for block in cache.block_pool.blocks[1:10]:
            policy.on_block_cached(block.block_id, block.block_hash_num_tokens)
        cache.block_pool.mwf_eviction_manager = policy
        saved = {key: (value.last_accessed, value.prefix_tokens)
                 for key, value in policy._metadata.items()}
        prefix, _, _ = cache.get_computed_blocks(request)
        before = self.snapshot(cache, manager, request)

        def select(count, *, pool):
            return pool.free_block_queue.get_all_free_blocks()[:count]

        with (
            patch.object(policy, "select_blocks", side_effect=select),
            patch.object(policy, "resync") as resync,
            patch.object(cache, "_publish_multiseg_metadata", side_effect=RuntimeError),
        ):
            with self.assertRaises(RuntimeError):
                cache.allocate_slots(request, 160, new_computed_blocks=prefix)
            resync.assert_called()
        self.assertEqual(self.snapshot(cache, manager, request), before)
        self.assertEqual(
            {key: (value.last_accessed, value.prefix_tokens)
             for key, value in policy._metadata.items()}, saved,
        )

    def test_failed_pool_restore_poison_is_not_reusable(self):
        cache, manager, request = self.fixture()
        pool = cache.block_pool
        with patch.object(pool.free_block_queue, "__init__", side_effect=RuntimeError):
            with self.assertRaisesRegex(RuntimeError, "recreate the block pool"):
                with pool.multiseg_transaction():
                    raise ValueError("publication")
        for operation in (
            lambda: pool.get_new_blocks(1),
            lambda: pool.touch([pool.blocks[1]]),
            lambda: pool.free_blocks([pool.blocks[1]]),
        ):
            with self.assertRaisesRegex(RuntimeError, "recreate the block pool"):
                operation()


class TestMultiSegOracleGateRegressions(unittest.TestCase):
    """Fixed CPU regressions derived from Oracle-MWF diagnostics."""

    fixture = TestMultiSegInstallation.fixture
    snapshot = TestMultiSegInstallation.snapshot

    def assert_r00061_fixed_layout(self, hit_start, expected_hits, expected_new):
        # Source/evidence-derived intervals, not a per-block GPU trace replay.
        logical_blocks, num_tokens = 6056, 96885
        cache, manager, request = self.fixture(
            count=0, pool_size=logical_blocks + 3
        )
        cache.max_model_len = num_tokens
        request.num_tokens = request.num_prompt_tokens = num_tokens
        request.block_hashes = [
            BlockHash(index.to_bytes(4, "big"))
            for index in range(num_tokens // 16)
        ]
        pool = cache.block_pool
        hit_blocks = {}
        for physical_id, index in enumerate(range(hit_start, 6025), 1):
            block = pool.blocks[physical_id]
            key = make_block_hash_with_group_id(request.block_hashes[index], 0)
            pool._insert_block_hash(key, block, (index + 1) * 16)
            hit_blocks[index] = (block, key)
        sentinels = pool.blocks[-2:]
        for index, block in enumerate(sentinels):
            key = make_block_hash_with_group_id(
                BlockHash(b"sentinel" + bytes([index])), 0
            )
            pool._insert_block_hash(key, block, 16)
        sentinel_keys = [block.block_hash for block in sentinels]
        prefix, prefix_tokens, _ = cache.get_computed_blocks(request)
        self.assertEqual(prefix_tokens, 0)
        self.assertEqual(len(request.continuum_multiseg_cached_blocks), expected_hits)
        plan = cache.coordinator.plan_multiseg_blocks(
            request, num_tokens, request.continuum_multiseg_cached_blocks
        )
        self.assertEqual(len(plan.block_ids), logical_blocks)
        self.assertEqual(len(plan.holes), expected_new)
        self.assertEqual(plan.free_need, logical_blocks)
        allocate = pool.get_new_blocks
        evict = pool._maybe_evict_cached_block
        evicted = []

        def observe_eviction(block):
            result = evict(block)
            if result:
                evicted.append(block.block_id)
            return result

        with (
            patch.object(pool, "get_new_blocks", wraps=allocate) as allocated,
            patch.object(pool, "_maybe_evict_cached_block",
                         side_effect=observe_eviction),
        ):
            result = cache.allocate_slots(
                request, num_tokens, new_computed_blocks=prefix
            )
        allocated.assert_called_once_with(expected_new)
        self.assertEqual(evicted, [])
        self.assertEqual(len(result.blocks[0]), logical_blocks)
        self.assertEqual(len(manager.new_block_ids), expected_new)
        self.assertEqual(len(set(manager.new_block_ids)), expected_new)
        self.assertEqual(
            request.continuum_suffix_computed_block_locs,
            [hit_start * 16, 6025 * 16, 0],
        )
        new_ids = set(manager.new_block_ids)
        for index, (block, key) in hit_blocks.items():
            self.assertIs(result.blocks[0][index], block)
            self.assertEqual(block.ref_cnt, 1)
            self.assertTrue(
                pool.cached_block_hash_to_block.contain(key, block.block_id)
            )
            self.assertNotIn(block.block_id, new_ids)
        self.assertEqual([block.block_hash for block in sentinels], sentinel_keys)
        self.assertTrue(all(block.ref_cnt == 0 for block in sentinels))
        self.assertEqual(pool.get_num_free_blocks(), 2)
        self.assertIsNone(result.blocks[0][-1].block_hash)
        self.assertEqual(set(manager.take_new_block_ids()), new_ids)
        self.assertEqual(manager.take_new_block_ids(), [])
        cache.free(request)
        free = pool.free_block_queue.get_all_free_blocks()
        self.assertEqual(len(free), logical_blocks + 2)
        self.assertEqual(len({block.block_id for block in free}), len(free))
        self.assertTrue(all(block.ref_cnt == 0 for block in pool.blocks[1:]))

    def test_r00061_fixed_mwf_interval_allocates_53_pages(self):
        self.assert_r00061_fixed_layout(22, 6003, 53)

    def test_r00061_fixed_oracle_interval_allocates_33_pages(self):
        self.assert_r00061_fixed_layout(2, 6023, 33)

    def test_capacity_retry_uses_current_shared_refs_without_mutating_failures(self):
        cache, manager, request = self.fixture(count=1, pool_size=4)
        pool = cache.block_pool
        withheld = pool.blocks[3]
        pool.touch([withheld])
        prefix, _, _ = cache.get_computed_blocks(request)
        before = self.snapshot(cache, manager, request)
        for _ in range(2):
            self.assertIsNone(
                cache.allocate_slots(request, 48, new_computed_blocks=prefix)
            )
            self.assertEqual(self.snapshot(cache, manager, request), before)
        hit = pool.blocks[1]
        pool.touch([hit])
        pool.free_blocks([withheld])
        result = cache.allocate_slots(request, 48, new_computed_blocks=prefix)
        self.assertEqual(len(result.blocks[0]), 3)
        self.assertIs(result.blocks[0][1], hit)
        self.assertEqual(hit.ref_cnt, 2)
        self.assertEqual(len(manager.new_block_ids), 2)
        self.assertNotIn(hit.block_id, manager.new_block_ids)
        cache.free(request)
        self.assertEqual(hit.ref_cnt, 1)
        pool.free_blocks([hit])
        self.assertEqual(pool.get_num_free_blocks(), 3)

    def test_cold_read_bypass_clears_sparse_candidates_before_lookup(self):
        cache, manager, request = self.fixture()
        cache.get_computed_blocks(request)
        self.assertTrue(request.continuum_multiseg_cached_blocks)
        request.skip_reading_prefix_cache = True
        before_refs = tuple(block.ref_cnt for block in cache.block_pool.blocks)
        with patch.object(manager, "discover_multiseg_hits") as discover:
            blocks, computed, uncached = cache.get_computed_blocks(request)
        discover.assert_not_called()
        self.assertEqual(blocks.get_block_ids(), ([],))
        self.assertEqual((computed, uncached), (0, 0))
        self.assertIsNone(request.continuum_multiseg_cached_blocks)
        self.assertEqual(request.continuum_suffix_computed_block_locs, [0])
        self.assertEqual(
            tuple(block.ref_cnt for block in cache.block_pool.blocks), before_refs
        )


def test_chunked_local_attention_get_num_blocks_to_allocate():
    block_size = 2
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(attention_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 0, 20 * block_size
        )
        == 15
    )


def test_predictor_matches_allocator_blocks_calculation_with_admission_cap():
    """In forward steps, `get_num_blocks_to_allocate` must return exactly what
    `allocate_new_blocks` will pull; otherwise `block_pool.get_new_blocks`
    raises `ValueError: Cannot get N free blocks from the pool`.
    """
    block_size = 2
    sliding_window = 8  # 4-block live window
    cap = sliding_window // block_size

    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
        max_admission_blocks_per_request=cap,
    )

    request_id = "req"
    total_computed = 0
    # Walk through request forward steps. Check num_blocks returned by
    # `get_num_blocks_to_allocate` matches what `allocate_new_blocks` pulls
    for num_tokens in (4, 8, 12, 16):
        predicted = manager.get_num_blocks_to_allocate(
            request_id=request_id,
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=total_computed,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_tokens,
        )
        new_blocks = manager.allocate_new_blocks(
            request_id, num_tokens=num_tokens, num_tokens_main_model=num_tokens
        )
        assert predicted == len(new_blocks), (
            f"num_tokens={num_tokens}: predictor returned {predicted} "
            f"but allocator pulled {len(new_blocks)}"
        )
        total_computed = num_tokens
