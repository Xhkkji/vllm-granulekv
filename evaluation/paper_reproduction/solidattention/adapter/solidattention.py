"""SolidAttention-style block selection on the shared sparse boundary."""

from __future__ import annotations

import base64
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

import vllm.envs as envs
from vllm.attention.ops.sparse_kv import (
    SparseKVDeviceSelection,
    build_page_representatives_from_paged_key_cache, )
from vllm.attention.ops.sparse_kv import (
    record_sparse_kv_selection, record_sparse_kv_selection_d2h_wait,
    record_sparse_kv_selection_tolist_call)
from vllm.core.custom_schedulers.hierarchical_io.sparse_policy import (
    SparseKVActualSelection, )

from ...common.selectors import ExplicitSelector, TailSelector


class SolidAttentionSelector:
    name = "solidattention"

    def __init__(self, block_budget: int = 0,
                 selections: Optional[Sequence[Sequence[int]]] = None) -> None:
        self._delegate = (ExplicitSelector(selections, self.name)
                          if selections is not None else
                          TailSelector(block_budget, "solidattention_proxy"))

    def select(self, num_layers: int, num_blocks: int):
        return self._delegate.select(num_layers, num_blocks)


@dataclass
class _AttentionSelectionState:
    buffer: torch.Tensor
    fixed_blocks: torch.Tensor
    candidate_blocks: torch.Tensor
    prefix_count: int
    num_blocks: int
    count: int
    mapping_token: int
    ready: bool


@dataclass(frozen=True)
class _PredictionRepresentativeState:
    logical_block_indices: Tuple[int, ...]
    representatives: torch.Tensor


@dataclass
class _ValidationState:
    mode: str
    metadata: dict[str, Any]
    replay: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    records_by_key: dict[tuple[int, int], dict[str, Any]] = field(
        default_factory=dict)
    current_layer_keys: dict[int, tuple[int, int]] = field(default_factory=dict)
    decode_step: int = 0
    last_layer_index: Optional[int] = None
    selector_reference_mismatches: int = 0
    replay_selection_mismatches: int = 0
    kv_fingerprint_mismatches: int = 0
    attention_output_mismatches: int = 0
    attention_output_max_rel_error: float = 0.0
    attention_output_min_cosine: float = 1.0
    all_block_attention_rel_error: float = 0.0
    all_block_attention_cosine: float = 1.0
    first_mismatch_step: Optional[int] = None
    first_mismatch_layer: Optional[int] = None


class SolidAttentionPolicy:
    """Minimal SolidAttention-style policy for the shared sparse bridge.

    The policy keeps fixed initial/local blocks and selects the remaining
    immutable prefix blocks with a GPU query/page score.  It owns only policy
    metadata; physical mappings, I/O requests and allocator state stay in the
    existing vLLM/GranuleKV integration.
    """

    name = "solidattention"
    _REPLAY_METADATA_KEYS = (
        "model",
        "prompt_sha256",
        "block_size",
        "prefix_blocks",
        "num_layers",
        "page_identity",
    )

    def __init__(self, init_blocks: int = 2, local_blocks: int = 2) -> None:
        if init_blocks < 0 or local_blocks < 0:
            raise ValueError("init_blocks and local_blocks must be non-negative")
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self._previous_queries: Dict[str, Dict[int, torch.Tensor]] = defaultdict(
            dict)
        self._representatives: Dict[
            str, Dict[int, Tuple[int, torch.Tensor]]] = defaultdict(dict)
        # Prediction metadata is keyed by the stable prefix identity.  The
        # resident-only attention path uses the request-keyed GPU cache above;
        # it never consults this control-plane state.
        self._page_index_keys: Dict[str, str] = {}
        self._prediction_representatives: Dict[
            str, Dict[int, _PredictionRepresentativeState]] = defaultdict(dict)
        self._seen: set[tuple[str, int]] = set()
        self._attention_selection_buffers: Dict[
            str, Dict[int, _AttentionSelectionState]] = defaultdict(dict)
        self._dynamic_selection_buffers: Dict[
            str, Dict[int, torch.Tensor]] = defaultdict(dict)
        self._selection_control_buffers: Dict[str, torch.Tensor] = {}
        self._validation_states: Dict[str, _ValidationState] = {}
        self._completed_validation_reports: Dict[str, dict[str, Any]] = {}

    @staticmethod
    def _normalize_validation_mode(mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in ("off", "all_blocks", "record", "replay"):
            raise ValueError(f"unsupported sparse validation mode: {mode!r}")
        return normalized

    def configure_validation(
        self,
        request_id: str,
        mode: str,
        metadata: Optional[Mapping[str, Any]] = None,
        replay_records: Optional[Sequence[Mapping[str, Any]]] = None,
        replay_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not request_id:
            raise ValueError("validation request_id must not be empty")
        normalized = self._normalize_validation_mode(mode)
        replay: dict[tuple[int, int], dict[str, Any]] = {}
        if normalized == "replay":
            if not replay_records:
                raise ValueError("replay validation requires selection records")
            current_metadata = dict(metadata or {})
            source_metadata = dict(replay_metadata or {})
            for key in self._REPLAY_METADATA_KEYS:
                if key not in current_metadata or key not in source_metadata:
                    raise ValueError(
                        f"replay validation metadata lacks required key: {key}")
                if current_metadata[key] != source_metadata[key]:
                    raise ValueError(
                        "sparse replay metadata mismatch: "
                        f"{key} expected={source_metadata[key]!r} "
                        f"actual={current_metadata[key]!r}")
            for raw in replay_records:
                record = dict(raw)
                decode_step = int(record["decode_step"])
                layer_index = int(record["layer_index"])
                key = (decode_step, layer_index)
                if key in replay:
                    raise ValueError(f"duplicate replay selection record: {key}")
                if decode_step < 0:
                    raise ValueError("replay decode_step must be non-negative")
                if not 0 <= layer_index < int(current_metadata["num_layers"]):
                    raise ValueError("replay layer_index is outside the model")
                if int(record["prefix_blocks"]) != int(
                        current_metadata["prefix_blocks"]):
                    raise ValueError(
                        "replay record prefix_blocks differs from metadata")
                selected = tuple(int(index) for index in
                                 record["selected_prefix_blocks"])
                if (selected != tuple(sorted(set(selected)))
                        or any(index < 0 or index >= int(
                            current_metadata["prefix_blocks"])
                               for index in selected)):
                    raise ValueError(
                        "replay selected prefix blocks must be sorted, unique, "
                        "and inside the immutable prefix")
                page_identity = record.get("page_identity")
                if (page_identity is not None
                        and page_identity != current_metadata["page_identity"]):
                    raise ValueError(
                        "sparse replay record uses a different page identity")
                replay[key] = record
        self._completed_validation_reports.pop(request_id, None)
        self._validation_states[request_id] = _ValidationState(
            mode=normalized,
            metadata=dict(metadata or {}),
            replay=replay,
        )

    def _validation_state(self, request_id: str) -> Optional[_ValidationState]:
        state = self._validation_states.get(request_id)
        if state is not None:
            return state
        default_mode = self._normalize_validation_mode(
            envs.VLLM_GRANULEKV_SPARSE_VALIDATION_MODE)
        if default_mode == "off":
            return None
        state = _ValidationState(mode=default_mode, metadata={})
        self._validation_states[request_id] = state
        return state

    @staticmethod
    def _validation_report_payload(state: _ValidationState) -> dict[str, Any]:
        replay_unconsumed = len(set(state.replay) - set(state.records_by_key))
        return {
            "mode": state.mode,
            "metadata": dict(state.metadata),
            "records": list(state.records),
            "selector_reference_mismatches":
            state.selector_reference_mismatches,
            "replay_selection_mismatches":
            state.replay_selection_mismatches,
            "replay_unconsumed_records": replay_unconsumed,
            "kv_fingerprint_mismatches": state.kv_fingerprint_mismatches,
            "attention_output_mismatches": state.attention_output_mismatches,
            "attention_output_max_rel_error":
            state.attention_output_max_rel_error,
            "attention_output_min_cosine":
            state.attention_output_min_cosine,
            "all_block_attention_rel_error":
            state.all_block_attention_rel_error,
            "all_block_attention_cosine": state.all_block_attention_cosine,
            "logit_rel_error": None,
            "greedy_token_match": None,
            "first_mismatch_step": state.first_mismatch_step,
            "first_mismatch_layer": state.first_mismatch_layer,
        }

    def validation_report(self, request_id: str,
                          reset: bool = False) -> dict[str, Any]:
        state = self._validation_states.get(request_id)
        if state is not None:
            report = self._validation_report_payload(state)
        else:
            report = dict(self._completed_validation_reports.get(
                request_id, {
                    "mode": "off",
                    "metadata": {},
                    "records": [],
                    "selector_reference_mismatches": 0,
                    "replay_selection_mismatches": 0,
                    "replay_unconsumed_records": 0,
                    "kv_fingerprint_mismatches": 0,
                    "attention_output_mismatches": 0,
                    "attention_output_max_rel_error": 0.0,
                    "attention_output_min_cosine": 1.0,
                    "all_block_attention_rel_error": 0.0,
                    "all_block_attention_cosine": 1.0,
                    "logit_rel_error": None,
                    "greedy_token_match": None,
                    "first_mismatch_step": None,
                    "first_mismatch_layer": None,
                }))
        if reset:
            self._validation_states.pop(request_id, None)
            self._completed_validation_reports.pop(request_id, None)
        return report

    @staticmethod
    def _mark_validation_mismatch(state: _ValidationState,
                                  step: int,
                                  layer_index: int) -> None:
        if state.first_mismatch_step is None:
            state.first_mismatch_step = step
            state.first_mismatch_layer = layer_index

    @staticmethod
    def _next_validation_key(state: _ValidationState,
                             layer_index: int) -> tuple[int, int]:
        previous = state.last_layer_index
        current_key = (state.decode_step, layer_index)
        if (previous is not None
                and (layer_index < previous
                     or (layer_index == previous
                         and current_key in state.records_by_key))):
            state.decode_step += 1
        state.last_layer_index = layer_index
        key = (state.decode_step, layer_index)
        state.current_layer_keys[layer_index] = key
        return key

    def _apply_validation_selection(
        self,
        request_id: str,
        layer_index: int,
        selected_prefix: torch.Tensor,
        num_prefix_blocks: int,
        num_blocks: int,
        block_budget: int,
        representative_count: int,
    ) -> torch.Tensor:
        state = self._validation_state(request_id)
        if state is None or state.mode == "off":
            return selected_prefix
        key = self._next_validation_key(state, layer_index)
        step, _ = key
        natural = self._selection_indices_to_host(selected_prefix)
        existing = state.records_by_key.get(key)
        if existing is not None:
            previous = tuple(int(index) for index in
                             existing["selected_prefix_blocks"])
            if previous != natural and state.mode == "record":
                state.replay_selection_mismatches += 1
                self._mark_validation_mismatch(state, step, layer_index)
            if previous == natural:
                return selected_prefix
            return torch.tensor(previous,
                                dtype=torch.long,
                                device=selected_prefix.device)
        chosen = natural
        if state.mode == "all_blocks":
            chosen = tuple(range(num_prefix_blocks))
        elif state.mode == "replay":
            record = state.replay.get((step, layer_index))
            if record is None:
                raise RuntimeError(
                    "missing sparse replay selection: "
                    f"request={request_id} step={step} layer={layer_index}")
            expected_prefix = int(record["prefix_blocks"])
            if expected_prefix != num_prefix_blocks:
                raise RuntimeError(
                    "sparse replay prefix mismatch: "
                    f"expected={expected_prefix} actual={num_prefix_blocks}")
            chosen = tuple(int(index)
                           for index in record["selected_prefix_blocks"])
            if natural != chosen:
                state.replay_selection_mismatches += 1
                self._mark_validation_mismatch(state, step, layer_index)
        if any(index < 0 or index >= num_prefix_blocks for index in chosen):
            raise RuntimeError(
                "validation selection is outside the immutable prefix")
        suffix = tuple(range(num_prefix_blocks, num_blocks))
        record = {
            "decode_step": step,
            "layer_index": layer_index,
            "prefix_blocks": num_prefix_blocks,
            "selected_prefix_blocks": list(chosen),
            "attention_blocks": list(chosen + suffix),
            "block_budget": block_budget,
            "page_identity": state.metadata.get("page_identity"),
            "representative_count": representative_count,
        }
        if state.metadata.get("capture_trace", False):
            state.records.append(record)
        state.records_by_key[key] = record
        if chosen == natural:
            return selected_prefix
        return torch.tensor(chosen,
                            dtype=torch.long,
                            device=selected_prefix.device)

    @staticmethod
    def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
        host = tensor.detach().contiguous().cpu()
        raw = host.view(torch.uint8).numpy().tobytes()
        return {
            "dtype": str(host.dtype).removeprefix("torch."),
            "shape": list(host.shape),
            "data_b64": base64.b64encode(raw).decode("ascii"),
        }

    @staticmethod
    def _tensor_from_payload(payload: Mapping[str, Any]) -> torch.Tensor:
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype_name = str(payload["dtype"])
        if dtype_name not in dtypes:
            raise ValueError(
                f"unsupported validation tensor dtype: {dtype_name}")
        raw = bytearray(base64.b64decode(str(payload["data_b64"])))
        return torch.frombuffer(raw, dtype=dtypes[dtype_name]).clone().reshape(
            tuple(int(value) for value in payload["shape"]))

    @staticmethod
    def _kv_fingerprint(key: torch.Tensor, value: torch.Tensor) -> str:
        digest = hashlib.sha256()
        for tensor in (key, value):
            host = tensor.detach().contiguous().cpu()
            digest.update(str(host.dtype).encode("ascii"))
            digest.update(repr(tuple(host.shape)).encode("ascii"))
            digest.update(host.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def record_validation_attention(
        self,
        request_id: str,
        layer_index: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attention_output: torch.Tensor,
        dense_attention_output: Optional[torch.Tensor] = None,
    ) -> None:
        state = self._validation_states.get(request_id)
        if state is None or state.mode == "off":
            return
        if not state.metadata.get("capture_layer_data", False):
            return
        key = state.current_layer_keys.get(layer_index)
        if key is None:
            raise RuntimeError(
                "validation attention has no matching selection record")
        step, _ = key
        record = state.records_by_key[key]
        fingerprints = {}
        if state.metadata.get("capture_kv_fingerprints", True):
            for logical_block in record["attention_blocks"]:
                physical_block = int(block_table[0, int(logical_block)].item())
                fingerprints[str(logical_block)] = self._kv_fingerprint(
                    key_cache[physical_block], value_cache[physical_block])
        output_payload = self._tensor_payload(attention_output)
        if dense_attention_output is not None:
            actual = attention_output.detach().float().cpu()
            dense = dense_attention_output.detach().float().cpu()
            difference = torch.linalg.vector_norm(actual - dense)
            denominator = torch.linalg.vector_norm(dense).clamp_min(1e-12)
            relative_error = float((difference / denominator).item())
            cosine = float(torch.nn.functional.cosine_similarity(
                actual.flatten(), dense.flatten(), dim=0).item())
            record["dense_attention_rel_error"] = relative_error
            record["dense_attention_cosine"] = cosine
            state.all_block_attention_rel_error = max(
                state.all_block_attention_rel_error, relative_error)
            state.all_block_attention_cosine = min(
                state.all_block_attention_cosine, cosine)
            if (state.mode == "all_blocks"
                    and not torch.allclose(actual,
                                           dense,
                                           rtol=2e-2,
                                           atol=2e-2)):
                state.attention_output_mismatches += 1
                self._mark_validation_mismatch(state, step, layer_index)
        if state.mode != "replay":
            record["kv_fingerprints"] = fingerprints
            record["attention_output"] = output_payload
            return

        expected = state.replay[key]
        expected_fingerprints = expected.get("kv_fingerprints")
        if expected_fingerprints is None:
            raise RuntimeError("replay trace lacks KV fingerprints")
        mismatches = sum(
            expected_fingerprints.get(name) != fingerprint
            for name, fingerprint in fingerprints.items())
        missing = set(expected_fingerprints) - set(fingerprints)
        if mismatches or missing:
            state.kv_fingerprint_mismatches += mismatches + len(missing)
            self._mark_validation_mismatch(state, step, layer_index)

        expected_output_payload = expected.get("attention_output")
        if expected_output_payload is None:
            raise RuntimeError("replay trace lacks attention output")
        expected_output = self._tensor_from_payload(expected_output_payload)
        actual_output = attention_output.detach().float().cpu()
        reference = expected_output.float()
        if actual_output.shape != reference.shape:
            state.attention_output_mismatches += 1
            self._mark_validation_mismatch(state, step, layer_index)
            return
        difference = torch.linalg.vector_norm(actual_output - reference)
        denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
        relative_error = float((difference / denominator).item())
        cosine = float(torch.nn.functional.cosine_similarity(
            actual_output.flatten(), reference.flatten(), dim=0).item())
        state.attention_output_max_rel_error = max(
            state.attention_output_max_rel_error, relative_error)
        state.attention_output_min_cosine = min(
            state.attention_output_min_cosine, cosine)
        if not torch.allclose(actual_output, reference, rtol=2e-3, atol=2e-3):
            state.attention_output_mismatches += 1
            self._mark_validation_mismatch(state, step, layer_index)

    def validation_requires_dense_reference(self, request_id: str) -> bool:
        state = self._validation_states.get(request_id)
        return bool(state is not None
                    and state.metadata.get("compare_dense_attention", False))

    @staticmethod
    def _normalize_query(query: torch.Tensor) -> torch.Tensor:
        if query.ndim == 3:
            if query.shape[0] != 1:
                raise ValueError("SolidAttention only supports batch size 1")
            query = query[0]
        if query.ndim != 2:
            raise ValueError("query must have shape [heads, head_dim]")
        return query

    def _selection_indices_to_host(self, selected: torch.Tensor) -> tuple[int, ...]:
        """Copy selection metadata explicitly at its CPU control dependency."""
        if not selected.is_cuda:
            return tuple(int(index) for index in selected)
        if not envs.VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE:
            record_sparse_kv_selection_tolist_call()
            return tuple(int(index) for index in selected.tolist())
        count = int(selected.shape[0])
        buffer = self._selection_control_buffers.get("default")
        if (buffer is None or buffer.numel() < count
                or buffer.dtype != selected.dtype):
            capacity = count if buffer is None else max(count, buffer.numel() * 2)
            # The policy may be called under vLLM's inference mode.  Keep the
            # reusable host control buffer as a normal tensor because later
            # D2H copies can run outside that mode.
            with torch.inference_mode(False):
                buffer = torch.empty(capacity,
                                     dtype=selected.dtype,
                                     pin_memory=True)
            self._selection_control_buffers["default"] = buffer
        stream = torch.cuda.current_stream(selected.device)
        with torch.cuda.stream(stream):
            buffer[:count].copy_(selected, non_blocking=True)
            ready = torch.cuda.Event(blocking=False, interprocess=False)
            ready.record(stream)
        start = time.perf_counter()
        ready.synchronize()
        record_sparse_kv_selection_d2h_wait(
            (time.perf_counter() - start) * 1000.0)
        return tuple(int(index) for index in buffer[:count])

    @staticmethod
    def _score_pages(query: torch.Tensor,
                     representatives: torch.Tensor) -> torch.Tensor:
        """Score pages using the representative envelope without host copies."""
        query = SolidAttentionPolicy._normalize_query(query)
        if representatives.ndim == 4:
            if representatives.shape[1] != 2:
                raise ValueError("representative envelope must have axis 2")
            kv_heads = representatives.shape[2]
            if query.shape[0] % kv_heads != 0:
                raise ValueError("representatives do not match query heads")
            lower, upper = representatives[:, 0], representatives[:, 1]
            queries_per_kv = query.shape[0] // kv_heads
            grouped_query = query.reshape(kv_heads, queries_per_kv,
                                          query.shape[-1])
            positive = grouped_query.clamp_min(0)
            negative = grouped_query.clamp_max(0)
            scores = (torch.einsum("kqd,pkd->kqp", positive, upper) +
                      torch.einsum("kqd,pkd->kqp", negative, lower))
            return scores.reshape(query.shape[0], representatives.shape[0])
        if representatives.ndim != 3:
            raise ValueError("representatives must have shape [pages, heads, dim]")
        if representatives.shape[1] <= 0 or query.shape[0] % representatives.shape[1] != 0:
            raise ValueError("representatives do not match query heads")
        kv_heads = representatives.shape[1]
        queries_per_kv = query.shape[0] // kv_heads
        grouped_query = query.reshape(kv_heads, queries_per_kv,
                                      query.shape[-1]).abs()
        scores = torch.einsum("kqd,pkd->kqp", grouped_query, representatives)
        return scores.reshape(query.shape[0], representatives.shape[0])

    def _fixed_and_candidates(self, prefix_count: int,
                              device: torch.device) -> tuple[torch.Tensor,
                                                              torch.Tensor]:
        init_end = min(self.init_blocks, prefix_count)
        local_start = max(init_end, prefix_count - self.local_blocks)
        init = torch.arange(init_end, dtype=torch.long, device=device)
        local = torch.arange(local_start, prefix_count, dtype=torch.long,
                             device=device)
        candidates = torch.arange(init_end, local_start, dtype=torch.long,
                                  device=device)
        return torch.cat((init, local)), candidates

    @staticmethod
    def _union_topk(scores: torch.Tensor, candidates: torch.Tensor,
                    budget: int) -> torch.Tensor:
        if candidates.numel() == 0 or budget <= 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        count = min(budget, candidates.numel())
        topk = torch.argsort(scores,
                             dim=-1,
                             descending=True,
                             stable=True)[:, :count]
        mask = torch.zeros(candidates.numel(),
                           dtype=torch.bool,
                           device=scores.device)
        mask.scatter_(0, topk.reshape(-1), True)
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        return candidates.index_select(0, selected)

    def _select_prefix(self, query: torch.Tensor,
                       representatives: torch.Tensor, prefix_count: int,
                       block_budget: int,
                       fixed_and_candidates: Optional[tuple[torch.Tensor,
                                                           torch.Tensor]] = None
                       ) -> torch.Tensor:
        prefix_count = min(prefix_count, representatives.shape[0])
        if prefix_count <= 0:
            return torch.empty(0, dtype=torch.long, device=query.device)
        if fixed_and_candidates is None:
            fixed, candidates = self._fixed_and_candidates(prefix_count,
                                                           query.device)
        else:
            fixed, candidates = fixed_and_candidates
        if candidates.numel() == 0:
            return fixed
        candidate_representatives = representatives.index_select(0, candidates)
        scores = self._score_pages(query, candidate_representatives)
        dynamic = self._union_topk(scores, candidates, block_budget)
        mask = torch.zeros(prefix_count, dtype=torch.bool, device=query.device)
        mask.scatter_(0, torch.cat((fixed, dynamic)), True)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _select_prefix_with_logical_indices(
        self,
        query: torch.Tensor,
        representatives: torch.Tensor,
        logical_block_indices: Tuple[int, ...],
        prefix_count: int,
        block_budget: int,
    ) -> torch.Tensor:
        """Select a prefix when metadata carries explicit logical indices.

        This helper is used by the control-plane prediction/check path.  The
        resident GPU selector keeps its existing dense tensor path untouched.
        """
        if representatives.shape[0] != len(logical_block_indices):
            raise ValueError(
                "SolidAttention representative count does not match logical "
                "block indices")
        if any(index < 0 for index in logical_block_indices) or any(
                left >= right for left, right in zip(
                    logical_block_indices, logical_block_indices[1:])):
            raise ValueError(
                "SolidAttention logical block indices must be sorted and "
                "unique")
        if any(index >= prefix_count for index in logical_block_indices):
            raise ValueError(
                "SolidAttention representative is outside immutable prefix")
        if query.device != representatives.device:
            query = query.to(device=representatives.device)

        prefix_count = max(0, prefix_count)
        if prefix_count == 0:
            return torch.empty(0, dtype=torch.long, device=query.device)
        init_end = min(self.init_blocks, prefix_count)
        local_start = max(init_end, prefix_count - self.local_blocks)
        fixed = torch.cat((
            torch.arange(init_end, dtype=torch.long, device=query.device),
            torch.arange(local_start,
                         prefix_count,
                         dtype=torch.long,
                         device=query.device),
        ))
        candidate_logical = tuple(
            index for index in logical_block_indices
            if init_end <= index < local_start)
        if not candidate_logical:
            return fixed
        candidate_positions = tuple(
            position for position, index in enumerate(logical_block_indices)
            if init_end <= index < local_start)
        positions = torch.tensor(candidate_positions,
                                  dtype=torch.long,
                                  device=representatives.device)
        candidates = torch.tensor(candidate_logical,
                                  dtype=torch.long,
                                  device=representatives.device)
        scores = self._score_pages(
            query, representatives.index_select(0, positions))
        dynamic = self._union_topk(scores, candidates, block_budget)
        mask = torch.zeros(prefix_count,
                           dtype=torch.bool,
                           device=representatives.device)
        mask.scatter_(0, torch.cat((fixed.to(mask.device), dynamic)), True)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _get_prediction_representatives(
        self,
        request_id: str,
        layer_index: int,
        num_prefix_blocks: int,
    ) -> Optional[tuple[Tuple[int, ...], torch.Tensor]]:
        page_index_key = self._page_index_keys.get(request_id, request_id)
        state = self._prediction_representatives.get(page_index_key,
                                                     {}).get(layer_index)
        if state is None:
            cached = self._representatives.get(page_index_key,
                                               {}).get(layer_index)
            if cached is None:
                return None
            cached_count, representatives = cached
            logical_block_indices = tuple(range(cached_count))
        else:
            logical_block_indices = state.logical_block_indices
            representatives = state.representatives
        positions = tuple(
            position for position, index in enumerate(logical_block_indices)
            if index < num_prefix_blocks)
        if not positions:
            return None
        if len(positions) != len(logical_block_indices):
            position_tensor = torch.tensor(positions,
                                           dtype=torch.long,
                                           device=representatives.device)
            representatives = representatives.index_select(0, position_tensor)
            logical_block_indices = tuple(logical_block_indices[position]
                                          for position in positions)
        return logical_block_indices, representatives

    def _cached_representatives(self, request_id: str, layer_index: int,
                                key_cache: torch.Tensor,
                                physical_block_ids: torch.Tensor,
                                prefix_count: int,
                                device: torch.device) -> torch.Tensor:
        cached_count, representatives = self._representatives[request_id].get(
            layer_index, (0, None))
        if cached_count > prefix_count:
            cached_count, representatives = 0, None
        if cached_count < prefix_count:
            new_ids = physical_block_ids[cached_count:prefix_count]
            new_representatives = build_page_representatives_from_paged_key_cache(
                key_cache, new_ids, output_device=device)
            representatives = (new_representatives
                               if representatives is None else
                               torch.cat((representatives, new_representatives)))
            cached_count = prefix_count
            self._representatives[request_id][layer_index] = (
                cached_count, representatives)
        if representatives is None:
            return torch.empty((0, 2, 0, 0), device=device)
        if representatives.device != device:
            representatives = representatives.to(device=device)
            self._representatives[request_id][layer_index] = (
                cached_count, representatives)
        return representatives[:prefix_count]

    @staticmethod
    def _mapping_token(physical_block_ids: torch.Tensor) -> int:
        # The resident decode mapping is append-only.  vLLM may reallocate
        # the block-table storage while growing it, even though existing
        # logical-to-physical entries are unchanged.  Use the logical view
        # length as the stable growth token; selected-list mode is disabled
        # for dynamic restore/remapping paths.
        return int(physical_block_ids.numel())

    def _prepare_device_selection(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        sequence_length: int,
        block_size: int,
        block_budget: int,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        query = self._normalize_query(query)
        if sequence_length <= 0 or block_size <= 0 or block_budget <= 0:
            raise ValueError("sequence and block parameters must be positive")
        num_blocks = (sequence_length + block_size - 1) // block_size
        if physical_block_ids.ndim != 1 or physical_block_ids.numel() < num_blocks:
            raise ValueError("physical block ids do not cover the sequence")
        prefix_count = max(0, num_blocks - 1)
        representatives = self._cached_representatives(
            request_id, layer_index, key_cache, physical_block_ids,
            prefix_count, query.device)
        return query, num_blocks, prefix_count, representatives

    def observe_query(self, request_id: str, layer_index: int,
                      query: torch.Tensor) -> None:
        self._previous_queries[request_id][layer_index] = (
            self._normalize_query(query).detach().clone())

    def register_page_representatives(
        self,
        request_id: str,
        layer_index: int,
        page_representatives: torch.Tensor,
        logical_block_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if not isinstance(page_representatives, torch.Tensor):
            raise TypeError("page_representatives must be a tensor")
        if page_representatives.ndim not in (3, 4):
            raise ValueError(
                "SolidAttention representatives must have 3 or 4 dimensions")
        logical = (tuple(range(page_representatives.shape[0]))
                   if logical_block_indices is None else
                   tuple(int(index) for index in logical_block_indices))
        if len(logical) != page_representatives.shape[0]:
            raise ValueError(
                "logical block indices must match representative pages")
        if any(index < 0 for index in logical) or any(
                left >= right for left, right in zip(logical, logical[1:])):
            raise ValueError(
                "SolidAttention logical block indices must be sorted and "
                "unique")
        representatives = page_representatives.detach()
        self._representatives[request_id][layer_index] = (
            len(logical), representatives)
        page_index_key = self._page_index_keys.get(request_id, request_id)
        self._prediction_representatives[page_index_key][layer_index] = (
            _PredictionRepresentativeState(logical, representatives))

    def bind_page_index_key(self, request_id: str, page_index_key: str) -> None:
        if not request_id or not page_index_key:
            raise ValueError("request_id and page_index_key must not be empty")
        # This mapping is only read by prediction/validation.  The resident
        # selected-list path remains request-local and does not touch it.
        self._page_index_keys[request_id] = page_index_key

    def seed_page_metadata_from_request(
        self,
        source_request_id: str,
        page_index_key: str,
        num_prefix_blocks: int,
    ) -> int:
        """Publish warmup representatives under a stable prefix identity.

        Working-set experiments intentionally disable GPU -> SSD writeback, so
        a metadata-only warmup is the source of SolidAttention representatives.
        The data path remains unchanged; this only makes that already-built
        control-plane metadata available to the subsequent restore request.
        """
        source = self._representatives.get(source_request_id, {})
        published = 0
        for layer_index, (_, representatives) in source.items():
            count = min(int(num_prefix_blocks), int(representatives.shape[0]))
            if count <= 0:
                continue
            logical = tuple(range(count))
            self._prediction_representatives[page_index_key][layer_index] = (
                _PredictionRepresentativeState(
                    logical, representatives[:count].detach()))
            published += 1
        return published

    def _select_blocks_device_impl(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        sequence_length: int,
        block_size: int,
        block_budget: int,
    ) -> torch.Tensor:
        query, num_blocks, prefix_count, representatives = (
            self._prepare_device_selection(
                request_id, layer_index, query, key_cache, physical_block_ids,
                sequence_length, block_size, block_budget))

        key = (request_id, layer_index)
        if key not in self._seen:
            self._seen.add(key)
            selected_prefix = torch.arange(prefix_count,
                                           dtype=torch.long,
                                           device=query.device)
            selected_prefix = self._apply_validation_selection(
                request_id, layer_index, selected_prefix, prefix_count,
                num_blocks, block_budget, int(representatives.shape[0]))
            suffix = torch.arange(prefix_count,
                                  num_blocks,
                                  dtype=torch.long,
                                  device=query.device)
            return torch.cat((selected_prefix, suffix))

        selected_prefix = self._select_prefix(query, representatives,
                                               prefix_count, block_budget)
        selected_prefix = self._apply_validation_selection(
            request_id, layer_index, selected_prefix, prefix_count, num_blocks,
            block_budget, int(representatives.shape[0]))
        suffix = torch.arange(prefix_count, num_blocks, dtype=torch.long,
                              device=query.device)
        return torch.cat((selected_prefix, suffix))

    def select_blocks_device(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        sequence_length: int,
        block_size: int,
        block_budget: int,
    ) -> torch.Tensor:
        return self._select_blocks_device_impl(
            request_id, layer_index, query, key_cache, physical_block_ids,
            sequence_length, block_size, block_budget)

    def select_attention_blocks_device(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        sequence_length: int,
        block_size: int,
        block_budget: int,
    ) -> SparseKVDeviceSelection:
        """Return a reusable GPU logical-block buffer for selected attention."""
        key = (request_id, layer_index)
        mapping_token = self._mapping_token(physical_block_ids)
        state = self._attention_selection_buffers[request_id].get(layer_index)
        if state is not None and state.mapping_token != mapping_token:
            # A longer logical view means that a new live block became
            # immutable.  Keep the request's warmup state and append only the
            # new representative; the selection itself is refreshed below.
            state = None

        query, num_blocks, prefix_count, representatives = (
            self._prepare_device_selection(
                request_id, layer_index, query, key_cache, physical_block_ids,
                sequence_length, block_size, block_budget))
        validation = self._validation_state(request_id)
        reuse = (envs.VLLM_GRANULEKV_SPARSE_SELECTION_REUSE_ENABLE
                 and (validation is None or validation.mode == "off"))
        if (reuse and state is not None and state.ready
                and state.prefix_count == prefix_count
                and state.num_blocks == num_blocks
                and state.buffer.device == query.device):
            record_sparse_kv_selection(refresh=False)
            return SparseKVDeviceSelection(state.buffer, state.count)

        if (state is not None and state.prefix_count == prefix_count
                and state.fixed_blocks.device == query.device):
            fixed_and_candidates = (state.fixed_blocks, state.candidate_blocks)
        else:
            fixed_and_candidates = self._fixed_and_candidates(
                prefix_count, query.device)

        if key not in self._seen:
            self._seen.add(key)
            selected_prefix = torch.arange(prefix_count,
                                           dtype=torch.long,
                                           device=query.device)
            selected_prefix = self._apply_validation_selection(
                request_id, layer_index, selected_prefix, prefix_count,
                num_blocks, block_budget, int(representatives.shape[0]))
            suffix = torch.arange(prefix_count,
                                  num_blocks,
                                  dtype=torch.long,
                                  device=query.device)
            selected = torch.cat((selected_prefix, suffix))
            ready = False
        else:
            selected_prefix = self._select_prefix(query, representatives,
                                                   prefix_count, block_budget,
                                                   fixed_and_candidates)
            selected_prefix = self._apply_validation_selection(
                request_id, layer_index, selected_prefix, prefix_count,
                num_blocks, block_budget, int(representatives.shape[0]))
            suffix = torch.arange(prefix_count, num_blocks, dtype=torch.long,
                                  device=query.device)
            selected = torch.cat((selected_prefix, suffix))
            ready = True
        record_sparse_kv_selection(refresh=True)
        count = selected.shape[0]
        if state is None or state.buffer.numel() < count or state.buffer.device != selected.device:
            capacity = count if state is None else max(count, state.buffer.numel() * 2)
            buffer = torch.empty(capacity,
                                 dtype=torch.int32,
                                 device=selected.device)
        else:
            buffer = state.buffer
        buffer[:count].copy_(selected)
        self._attention_selection_buffers[request_id][layer_index] = (
            _AttentionSelectionState(buffer, fixed_and_candidates[0],
                                     fixed_and_candidates[1], prefix_count,
                                     num_blocks, count, mapping_token, ready))
        return SparseKVDeviceSelection(buffer, count)

    def select_blocks(self, request_id: str, layer_index: int,
                      query: torch.Tensor,
                      page_representatives: Optional[torch.Tensor],
                      num_blocks: int,
                      block_budget: int) -> tuple[int, ...]:
        query = self._normalize_query(query)
        if num_blocks <= 0 or block_budget <= 0:
            raise ValueError("num_blocks and block_budget must be positive")
        representatives = page_representatives
        if representatives is None or representatives.numel() == 0:
            representatives = self._representatives[request_id].get(
                layer_index, (0, None))[1]
        if representatives is None:
            return tuple(range(num_blocks))
        key = (request_id, layer_index)
        if key not in self._seen:
            self._seen.add(key)
            return tuple(range(num_blocks))
        prefix_count = max(0, min(num_blocks - 1, representatives.shape[0]))
        selected = self._select_prefix(query, representatives, prefix_count,
                                       block_budget)
        selected = self._apply_validation_selection(
            request_id, layer_index, selected, prefix_count, num_blocks,
            block_budget, int(representatives.shape[0]))
        suffix = torch.arange(prefix_count, num_blocks, dtype=torch.long,
                              device=selected.device)
        return self._selection_indices_to_host(torch.cat((selected, suffix)))

    def select_blocks_for_residency_check(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        num_blocks: int,
        block_budget: int,
    ) -> tuple[int, ...]:
        """Select the current working set without request warmup.

        Dynamic restore uses this control-plane check to detect a prediction
        miss before the compact compatibility consumer reads KV.  It is not
        used by resident-only selected-list attention.
        """
        if num_blocks <= 0 or block_budget <= 0:
            raise ValueError("num_blocks and block_budget must be positive")
        prefix_count = max(0, num_blocks - 1)
        metadata = self._get_prediction_representatives(
            request_id, layer_index, prefix_count)
        if metadata is None:
            raise RuntimeError(
                "SolidAttention prediction validation lacks prefix metadata: "
                f"request={request_id} layer={layer_index}")
        logical, representatives = metadata
        selected_prefix = self._select_prefix_with_logical_indices(
            self._normalize_query(query), representatives, logical,
            prefix_count, block_budget)
        suffix = torch.arange(prefix_count,
                              num_blocks,
                              dtype=torch.long,
                              device=selected_prefix.device)
        return self._selection_indices_to_host(
            torch.cat((selected_prefix, suffix)))

    def _select_actual_prefix_tensor(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        num_prefix_blocks: int,
        block_budget: int,
        output_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if num_prefix_blocks <= 0 or block_budget <= 0:
            raise ValueError(
                "num_prefix_blocks and block_budget must be positive")
        metadata = self._get_prediction_representatives(
            request_id, layer_index, num_prefix_blocks)
        if metadata is None:
            raise RuntimeError(
                "SolidAttention actual selection lacks prefix metadata: "
                f"request={request_id} layer={layer_index}")
        logical, representatives = metadata
        if output_device is not None and representatives.device != output_device:
            representatives = representatives.to(device=output_device)
            query = query.to(device=output_device)
            page_index_key = self._page_index_keys.get(request_id, request_id)
            self._prediction_representatives[page_index_key][layer_index] = (
                _PredictionRepresentativeState(logical, representatives))
        return self._select_prefix_with_logical_indices(
            self._normalize_query(query), representatives, logical,
            num_prefix_blocks, block_budget)

    def select_actual_prefix_blocks(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        num_prefix_blocks: int,
        block_budget: int,
    ) -> tuple[int, ...]:
        """Select only immutable prefix blocks for the current query."""
        selected = self._select_actual_prefix_tensor(
            request_id, layer_index, query, num_prefix_blocks, block_budget)
        metadata = self._get_prediction_representatives(
            request_id, layer_index, num_prefix_blocks)
        representative_count = 0 if metadata is None else len(metadata[0])
        selected = self._apply_validation_selection(
            request_id, layer_index, selected, num_prefix_blocks,
            num_prefix_blocks, block_budget, representative_count)
        return self._selection_indices_to_host(selected)

    def select_actual_blocks_device(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        num_prefix_blocks: int,
        num_blocks: int,
        block_budget: int,
    ) -> SparseKVActualSelection:
        """Build one current-query selection for correction and attention."""
        if num_blocks < num_prefix_blocks:
            raise ValueError("num_blocks must cover the immutable prefix")
        selected_prefix = self._select_actual_prefix_tensor(
            request_id, layer_index, query, num_prefix_blocks, block_budget,
            output_device=query.device)
        metadata = self._get_prediction_representatives(
            request_id, layer_index, num_prefix_blocks)
        representative_count = 0 if metadata is None else len(metadata[0])
        selected_prefix = self._apply_validation_selection(
            request_id, layer_index, selected_prefix, num_prefix_blocks,
            num_blocks, block_budget, representative_count)
        if not selected_prefix.is_cuda:
            raise RuntimeError(
                "dynamic selected-list attention requires CUDA metadata")
        suffix = torch.arange(num_prefix_blocks,
                              num_blocks,
                              dtype=torch.long,
                              device=selected_prefix.device)
        selected = torch.cat((selected_prefix, suffix))
        count = int(selected.shape[0])
        buffer = self._dynamic_selection_buffers[request_id].get(layer_index)
        if (buffer is None or buffer.device != selected.device
                or buffer.numel() < count):
            capacity = count if buffer is None else max(count,
                                                        buffer.numel() * 2)
            buffer = torch.empty(capacity,
                                 dtype=torch.int32,
                                 device=selected.device)
            self._dynamic_selection_buffers[request_id][layer_index] = buffer
        buffer[:count].copy_(selected)
        prefix_blocks = self._selection_indices_to_host(selected_prefix)
        return SparseKVActualSelection(
            prefix_blocks=prefix_blocks,
            attention=SparseKVDeviceSelection(
                buffer, count, prefix_count=int(selected_prefix.shape[0])),
            prefix_count=int(selected_prefix.shape[0]),
        )

    def predict_restore_blocks(self, request_id: str, layer_index: int,
                               num_prefix_blocks: int,
                               block_budget: int) -> Optional[Tuple[int, ...]]:
        history = self._previous_queries.get(request_id, {}).get(layer_index)
        if history is None:
            return None
        metadata = self._get_prediction_representatives(
            request_id, layer_index, num_prefix_blocks)
        if metadata is None:
            return None
        logical, representatives = metadata
        guard_blocks = envs.VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS
        if guard_blocks < 0:
            raise ValueError("prediction guard blocks must be non-negative")
        selected = self._select_prefix_with_logical_indices(
            history, representatives, logical, num_prefix_blocks,
            block_budget + guard_blocks)
        return self._selection_indices_to_host(selected)

    def discard(self, request_id: str) -> None:
        validation = self._validation_states.pop(request_id, None)
        if (validation is not None
                and validation.metadata.get("retain_report", False)):
            self._completed_validation_reports[request_id] = (
                self._validation_report_payload(validation))
        page_index_key = self._page_index_keys.pop(request_id, None)
        self._previous_queries.pop(request_id, None)
        self._representatives.pop(request_id, None)
        # Stable-prefix metadata is shared by requests using the same page
        # key.  Request-local metadata is always removed; the stable entry is
        # retained until a later prefix registration replaces it.
        if page_index_key is None or page_index_key == request_id:
            self._prediction_representatives.pop(request_id, None)
        self._attention_selection_buffers.pop(request_id, None)
        self._dynamic_selection_buffers.pop(request_id, None)
        self._seen = {key for key in self._seen if key[0] != request_id}
