# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One user-operated GPU gate worker; compare its artifacts in the gate runner.

Uses ordinary installed vLLM imports and public full-vocabulary raw logits.
It observes real BlockPool methods without replacing candidate indices/scores.
No binaries are copied, built, or borrowed from another installation.
"""

import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from collections import Counter

from run_mwf_gate import require_origin, validate_result_root


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def provenance(repo, policy):
    names = ["vllm", "vllm._C", "vllm.v1.core.block_pool",
             "vllm.v1.core.mwf_eviction", "vllm.v1.core.kv_cache_manager"]
    if policy == "mwf":
        names.append("vllm._mwf_candidate_index")
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require_origin(path, repo)
        modules[name] = {"path": str(path), "sha256": digest(path)}
    return modules


def native_sources(repo, torch):
    origins = [(name, getattr(module, "__file__", None))
               for name, module in list(sys.modules.items())]
    origins += [("torch.ops", path) for path in torch.ops.loaded_libraries]
    libraries = {}
    for name, filename in origins:
        if not filename or not str(filename).endswith(".so"):
            continue
        path = Path(filename).resolve()
        is_vllm = name.startswith("vllm.") or "vllm" in path.parts
        if is_vllm:
            require(path.is_relative_to(repo), f"Wrong vLLM native library: {path}")
        if str(path) not in libraries:
            libraries[str(path)] = {"sha256": digest(path), "modules": [],
                                    "vllm": is_vllm}
        libraries[str(path)]["vllm"] |= is_vllm
        libraries[str(path)]["modules"].append(name)
    return libraries


class PoolObservation:
    """Count successful physical cached-page reuse through the real methods."""

    def __init__(self, pool):
        self.pool = pool
        self.evictions = []
        self.mwf_allocations = []
        self.restores = []
        self.case = "startup"
        self.wrap("_get_new_blocks_mwf", self.allocation)
        self.wrap("_maybe_evict_cached_block", self.eviction)

    def wrap(self, name, observe):
        original = getattr(self.pool, name)
        present = name in self.pool.__dict__
        previous = self.pool.__dict__.get(name)
        self.restores.append((name, present, previous))
        setattr(self.pool, name, lambda *a, **k: observe(original, *a, **k))

    def allocation(self, original, count):
        result = original(count)
        if count:
            self.mwf_allocations.append({
                "case": self.case, "count": count,
                "block_ids": [block.block_id for block in result],
            })
        return result

    def eviction(self, original, block):
        old_hash = block.block_hash
        old_ref = block.ref_cnt
        prefix = block.block_hash_num_tokens
        result = original(block)
        if old_hash is not None and old_ref == 0 and result:
            self.evictions.append({
                "case": self.case, "block_id": block.block_id,
                "old_hash": old_hash.hex(), "prefix_tokens": prefix,
                "ref_before": old_ref, "hash_cleared": block.block_hash is None,
            })
            require(block.block_hash is None, "Eviction retained its primary hash")
        return result

    def audit(self, idle=False):
        pool = self.pool
        queue = pool.free_block_queue
        count = pool.get_num_free_blocks()
        free = list(itertools.islice(queue.iter_blocks_after(None),
                                    pool.num_gpu_blocks + 1))
        ids = [block.block_id for block in free]
        last = free[-1] if free else queue.fake_free_list_head
        require(len(ids) == len(set(ids)) == count,
                "Free queue identity/count mismatch")
        require(last.next_free_block is queue.fake_free_list_tail, "Invalid queue tail")
        expected = {b.block_id for b in pool.blocks if not b.is_null and b.ref_cnt == 0}
        require(set(ids) == expected, "Free queue differs from ref0 physical pages")
        require(all(b.pool is pool and pool.blocks[b.block_id] is b for b in free),
                "Free queue contains noncanonical pages")
        require(all(b.ref_cnt >= 0 for b in pool.blocks if not b.is_null),
                "Negative physical-page reference count")
        manager = pool.mwf_eviction_manager
        if manager is not None:
            require(manager._index.size() == count, "MWF/free queue size mismatch")
            hashed = {b.block_id for b in pool.blocks if b.block_hash is not None}
            require(set(manager._metadata) == hashed, "Stale/missing MWF metadata")
        if idle:
            require(count == pool.num_gpu_blocks - 1,
                    "Idle engine leaked page references")

    def close(self):
        for name, present, previous in reversed(self.restores):
            if present:
                setattr(self.pool, name, previous)
            else:
                delattr(self.pool, name)


class Cases:
    def __init__(self, llm, output, observer):
        import numpy as np
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        self.np, self.params_type = np, SamplingParams
        self.output_kind = RequestOutputKind.CUMULATIVE
        self.llm, self.engine, self.output = llm, llm.llm_engine, output
        self.scheduler = self.engine.engine_core.engine_core.scheduler
        self.pool, self.observer = self.scheduler.kv_cache_manager.block_pool, observer
        self.vocab = self.engine.model_config.get_vocab_size()
        self.latest, self.records, self.reference_pairs, self.ids = {}, [], [], {}
        self.coverage = {}
        tokenizer = llm.get_tokenizer()
        body = "A database stores records in pages. A cache reuses matching data. "
        body *= 100

        def encode(label):
            ids = tokenizer.encode(f"{label}: {body}", add_special_tokens=True)
            require(len(ids) >= 256, "Tokenizer input is too short")
            return ids[:256]

        self.a = encode("ALPHA")
        self.pressure = [encode(label)
                         for label in ("BRAVO", "CHARLIE", "DELTA", "ECHO")]
        prefixes = {tuple(ids[:16]) for ids in [self.a, *self.pressure]}
        require(len(prefixes) == 5, "Pressure prompts must have distinct first pages")
        self.b = self.pressure[0]
        self.shared = self.a[:192] + encode("FOXTROT")[:64]
        write_json(output / "inputs.json", {"a": self.a, "b": self.b,
                   "shared": self.shared, "pressure": self.pressure})

    def reset(self):
        require(not self.engine.has_unfinished_requests(),
                "Reset helper requires idle engine")
        require(self.llm.reset_prefix_cache(), "Idle prefix reset was refused")
        self.observer.audit(idle=True)
        require(all(b.block_hash is None for b in self.pool.blocks),
                "Reset retained hashes")

    def add(self, name, prompt, count):
        params = self.params_type(
            temperature=0.0, top_p=1.0, top_k=-1, seed=0, max_tokens=count,
            ignore_eos=True, repetition_penalty=1.0, presence_penalty=0.0,
            frequency_penalty=0.0, logprobs=-1, detokenize=False,
            output_kind=self.output_kind,
        )
        internal = self.engine.add_request(name, {"prompt_token_ids": prompt}, params)
        self.ids[name] = internal
        return {"name": name, "id": internal, "prompt": prompt, "count": count,
                "request": self.scheduler.requests[internal]}

    def step(self):
        for result in self.engine.step():
            internal = self.ids.get(result.request_id, result.request_id)
            self.latest[internal] = result
        self.observer.audit()
        managers = self.scheduler.kv_cache_manager.coordinator.single_type_managers
        owners = Counter(b.block_id for m in managers
                         for blocks in m.req_to_blocks.values()
                         for b in blocks if not b.is_null)
        require(all(b.ref_cnt == owners[b.block_id]
                    for b in self.pool.blocks if not b.is_null),
                "Request tables disagree with physical-page reference counts")

    def finish(self):
        for _ in range(4096):
            if not self.engine.has_unfinished_requests():
                self.observer.audit(idle=True)
                return
            self.step()
        raise RuntimeError("Request exceeded 4096 engine steps")

    def save(self, item, partial=False):
        np = self.np
        result = self.latest.pop(item["id"], None)
        require(result is not None and len(result.outputs) == 1,
                "Missing request output")
        require(list(result.prompt_token_ids) == item["prompt"],
                "Prompt token IDs changed")
        completion = result.outputs[0]
        tokens, rows = list(completion.token_ids), completion.logprobs
        require(tokens and rows is not None and len(rows) == len(tokens),
                "Missing or misaligned public raw-logit rows")
        require(partial or len(tokens) == item["count"], "Incomplete generation")
        logits = np.empty((len(tokens), self.vocab), dtype=np.float32)
        for i, row in enumerate(rows):
            require(len(row) == self.vocab, "Public logits omit vocabulary entries")
            ids = np.fromiter(row.keys(), dtype=np.int64, count=self.vocab)
            require(int(ids.min()) == 0 and int(ids.max()) == self.vocab - 1,
                    "Unexpected vocabulary token IDs")
            logits[i, ids] = np.fromiter((value.logprob for value in row.values()),
                                        dtype=np.float32, count=self.vocab)
        require(np.isfinite(logits).all(), "Non-finite raw logits")
        require(np.argmax(logits, axis=1).tolist() == tokens,
                "Raw logits/greedy mismatch")
        np.savez_compressed(self.output / f"{item['name']}.npz", logits=logits,
                            token_ids=np.asarray(tokens, dtype=np.int64),
                            prompt_token_ids=np.asarray(item["prompt"], dtype=np.int64))
        require(isinstance(result.num_cached_tokens, int),
                "Missing cache-hit accounting")
        record = {"name": item["name"], "token_ids": tokens, "partial": partial,
                  "cached_tokens": result.num_cached_tokens,
                  "num_preemptions": item["request"].num_preemptions}
        self.records.append(record)
        write_json(self.output / "records.json", self.records)
        print("CASE", item["name"], "tokens", len(tokens),
              "cached", result.num_cached_tokens,
              "preemptions", record["num_preemptions"], flush=True)
        return record

    def single(self, name, prompt, count=8, clear=True):
        if clear:
            self.reset()
        self.observer.case = name
        item = self.add(name, prompt, count)
        self.finish()
        return self.save(item)

    def basic(self):
        cold = self.single("a_cold", self.a)
        repeat = self.single("a_repeat", self.a, clear=False)
        require(cold["cached_tokens"] == 0, "Cold prompt hit a cache")
        require(repeat["cached_tokens"] >= 240,
                "Repeated prompt did not hit its prefix")
        for i, ids in enumerate(self.pressure):
            self.single(f"pressure_{i}", ids, clear=False)
        after = self.single("a_after_pressure", self.a, clear=False)
        require(after["cached_tokens"] < repeat["cached_tokens"],
                "NOT_COVERED: pressure did not invalidate the original prefix")
        shared = self.single("shared_prefix", self.shared, clear=False)
        require(shared["cached_tokens"] >= 192, "Shared prefix did not hit")
        final = self.single("a_after_reset", self.a)
        require(final["cached_tokens"] == 0, "Idle reset retained a cache hit")
        self.reference_pairs = [["a_cold", name] for name in
                                ("a_repeat", "a_after_pressure", "a_after_reset")]

    def shared_cancel(self):
        self.single("ref_a", self.a, 16)
        self.single("ref_shared", self.shared, 16)
        self.single("shared_warm", self.a)
        self.observer.case = "shared_pair"
        items = [self.add("shared_a", self.a, 16),
                 self.add("shared_b", self.shared, 16)]
        self.step()
        shared_ids = [b.block_id for b in self.pool.blocks
                      if not b.is_null and b.ref_cnt >= 2]
        require(shared_ids, "NOT_COVERED: no simultaneously shared physical pages")
        self.coverage["simultaneously_shared_blocks"] = shared_ids
        before = [(b.block_hash, b.ref_cnt) for b in self.pool.blocks]
        require(not self.llm.reset_prefix_cache(), "Busy reset was accepted")
        require(before == [(b.block_hash, b.ref_cnt) for b in self.pool.blocks],
                "Refused reset changed page hashes/references")
        self.observer.audit()
        self.coverage["busy_reset_refused"] = True
        self.finish()
        for item, ref in zip(items, ("ref_a", "ref_shared")):
            self.save(item)
            self.reference_pairs.append([ref, item["name"]])

        self.single("cancel_warm", self.a)
        self.observer.case = "running_cancel"
        cancelled = self.add("cancel_partial", self.a, 16)
        survivor = self.add("cancel_survivor", self.shared, 16)
        self.step()
        self.step()
        require(cancelled["request"].num_output_tokens == 2
                and cancelled["request"].status.name == "RUNNING",
                "NOT_COVERED: running cancellation was not after two tokens")
        self.save(cancelled, partial=True)
        self.engine.abort_request([cancelled["id"]], internal=True)
        require(cancelled["id"] not in self.scheduler.requests,
                "Running abort was not applied")
        self.coverage["running_cancelled_after_tokens"] = 2
        self.finish()
        self.save(survivor)
        self.reference_pairs += [["ref_a", "cancel_partial"],
                                 ["ref_shared", "cancel_survivor"]]

        self.single("queue_warm", self.a)
        self.observer.case = "waiting_cancel"
        items = [self.add("queue_a", self.a, 16),
                 self.add("queue_shared", self.shared, 16)]
        queued = self.add("cancel_waiting", self.b, 16)
        self.step()
        require(queued["request"].status.name == "WAITING"
                and queued["request"].num_computed_tokens == 0
                and queued["request"].num_output_tokens == 0,
                "NOT_COVERED: third request was not waiting")
        require(queued["id"] not in self.latest, "Waiting request produced output")
        self.engine.abort_request([queued["id"]], internal=True)
        require(queued["id"] not in self.scheduler.requests,
                "Waiting abort was not applied")
        self.coverage["waiting_cancelled_before_execution"] = True
        self.finish()
        for item, ref in zip(items, ("ref_a", "ref_shared")):
            self.save(item)
            self.reference_pairs.append([ref, item["name"]])
        final = self.single("after_cancel_reset", self.a, 16)
        require(final["cached_tokens"] == 0,
                "Cancellation/reset retained a cache hit")
        self.reference_pairs.append(["ref_a", "after_cancel_reset"])
        for i, ids in enumerate(self.pressure):
            self.single(f"eviction_probe_{i}", ids, clear=False)

    def preemption(self):
        self.single("ref_a", self.a, 96)
        self.single("ref_b", self.b, 96)
        self.reset()
        self.observer.case = "preemption_pair"
        items = [self.add("preempt_a", self.a, 96), self.add("preempt_b", self.b, 96)]
        self.finish()
        for item, ref in zip(items, ("ref_a", "ref_b")):
            self.save(item)
            self.reference_pairs.append([ref, item["name"]])
        count = sum(item["request"].num_preemptions for item in items)
        require(count > 0, "NOT_COVERED: capacity pressure did not cause preemption")
        self.coverage["actual_preemptions"] = count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", required=True, choices=("lru", "mwf"))
    parser.add_argument("--suite", required=True,
                        choices=("basic", "shared_cancel", "preemption"))
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--attention-backend", default="FLASHINFER",
                        choices=("FLASHINFER",))
    args = parser.parse_args()
    args.repo, args.model, args.output = (p.resolve() for p in
                                        (args.repo, args.model, args.output))
    require(args.output.is_dir(), "--output must be an existing worker directory")
    validate_result_root(args.output)
    require(not (args.output / "status.json").exists(),
            "Worker output already contains a run")
    require((args.model / "config.json").is_file(),
            "Explicit local model/config.json is missing")
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
    llm = observer = None
    write_json(args.output / "status.json", {"passed": None, "status": "starting"})
    try:
        import torch
        import vllm
        from vllm import LLM
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        modules = provenance(args.repo, args.policy)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "Gate requires exactly one visible CUDA GPU")
        parameters = dict(
            dtype=args.dtype, kv_cache_dtype=args.dtype, tensor_parallel_size=1,
            pipeline_parallel_size=1, max_model_len=512, max_num_batched_tokens=512,
            max_num_seqs=1 if args.suite == "basic" else 2, block_size=16,
            num_gpu_blocks_override=40, gpu_memory_utilization=0.35,
            enforce_eager=True, enable_prefix_caching=True,
            enable_chunked_prefill=False,
            async_scheduling=False, seed=0, max_logprobs=-1, logprobs_mode="raw_logits",
            attention_config={"backend": args.attention_backend},
            prefix_cache_eviction_policy=args.policy,
            mwf_minexp_a=0.0, mwf_minexp_b=-19.15349,
            mwf_minexp_alpha=1915.332, mwf_minexp_beta=47.63942,
        )
        llm = LLM(model=str(args.model), **parameters)
        engine = llm.llm_engine
        core = engine.engine_core.engine_core
        pool = core.scheduler.kv_cache_manager.block_pool
        config = engine.vllm_config
        groups = core.scheduler.kv_cache_manager.kv_cache_config.kv_cache_groups
        require(len(groups) == 1 and type(groups[0].kv_cache_spec) is FullAttentionSpec,
                "Gate requires one ordinary FullAttention KV group")
        spec = groups[0].kv_cache_spec
        require(spec.sliding_window is None and spec.attention_chunk_size is None
                and not spec.non_causal, "Gate requires ordinary causal attention")
        require(engine.model_config.get_head_size() == 128,
                "Gate requires head_dim=128")
        require(not config.use_v2_model_runner,
                "Controlled gate requires Model Runner V1")
        require(config.model_config.enforce_eager
                and not config.scheduler_config.async_scheduling,
                "Gate requires synchronous eager execution")
        require(config.model_config.logprobs_mode == "raw_logits", "Wrong logit mode")
        require(config.model_config.dtype == getattr(torch, args.dtype),
                "Wrong model dtype")
        require(config.attention_config.backend.name == "FLASHINFER",
                "Wrong attention backend")
        require(pool.num_gpu_blocks == 40 and pool.enable_caching,
                "Wrong GPU KV capacity")
        require((pool.mwf_eviction_manager is not None) == (args.policy == "mwf"),
                "Configured policy is not attached to the real GPU pool")
        runner = core.model_executor.driver_worker.worker.model_runner
        attention_layers = {
            name: {
                "backend": layer.get_attn_backend().get_name(),
                "impl": (f"{type(layer.impl).__module__}."
                         f"{type(layer.impl).__qualname__}"),
            }
            for name, layer in runner.get_model().named_modules()
            if isinstance(layer, Attention)
        }
        require(attention_layers and all(
            layer["backend"] == "FLASHINFER" for layer in attention_layers.values()
        ), "Actual model attention layers must all use FlashInfer")
        if pool.mwf_eviction_manager is not None:
            require(pool.mwf_eviction_manager._params == {
                f"minexp_{name}": parameters[f"mwf_minexp_{name}"]
                for name in ("a", "b", "alpha", "beta")
            }, "MWF parameters did not reach the real pool manager")
        flashinfer = importlib.import_module("flashinfer")
        manifest = {
            "scope": "GPU worker coverage/data capture; comparison is separate",
            "policy": args.policy, "suite": args.suite, "parameters": parameters,
            "repo": str(args.repo), "model": str(args.model), "python": sys.executable,
            "head": subprocess.check_output(
                ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                text=True).strip(),
            "git_status": subprocess.check_output(
                ["git", "-C", str(args.repo), "status", "--porcelain"], text=True),
            "modules": modules, "driver_sha256": digest(__file__),
            "model_config_sha256": digest(args.model / "config.json"),
            "vllm_version": vllm.__version__, "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
            "flashinfer_version": flashinfer.__version__,
            "flashinfer_path": flashinfer.__file__,
            "runner": f"{type(runner).__module__}.{type(runner).__qualname__}",
            "attention_layers": attention_layers,
            "use_v2_model_runner": config.use_v2_model_runner,
        }
        write_json(args.output / "manifest.json", manifest)
        observer = PoolObservation(pool)
        observer.audit(idle=True)
        cases = Cases(llm, args.output, observer)
        getattr(cases, args.suite)()
        observer.audit(idle=True)
        require(observer.evictions, "NOT_COVERED: no real ref0 cached-page eviction")
        require(bool(observer.mwf_allocations) == (args.policy == "mwf"),
                "NOT_COVERED: real MWF allocation path did not match policy")
        if args.policy == "lru":
            require("vllm._mwf_candidate_index" not in sys.modules,
                    "LRU loaded MWF native")
        require(provenance(args.repo, args.policy) == modules,
                "Loaded module sources changed")
        manifest["loaded_native_libraries"] = native_sources(args.repo, torch)
        write_json(args.output / "manifest.json", manifest)
        coverage = dict(cases.coverage, passed=True,
                        reference_pairs=cases.reference_pairs,
                        cached_evictions=len(observer.evictions),
                        mwf_allocations=len(observer.mwf_allocations))
        write_json(args.output / "coverage.json", coverage)
        write_json(args.output / "status.json", {"passed": True, "status": "finished",
                   "scope": "worker coverage and full-vocabulary raw-logit capture"})
        print("MWF_GPU_WORKER_PASS", args.suite, args.policy, flush=True)
    except Exception as error:
        write_json(args.output / "status.json", {"passed": False, "status": "failed",
                   "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if observer is not None:
            write_json(args.output / "observations.json", {
                       "evictions": observer.evictions,
                       "mwf_allocations": observer.mwf_allocations})
            observer.close()
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as error:
                write_json(args.output / "status.json", {
                    "passed": False, "status": "failed", "phase": "shutdown",
                    "error": str(error), "traceback": traceback.format_exc(),
                })
                raise


if __name__ == "__main__":
    main()
