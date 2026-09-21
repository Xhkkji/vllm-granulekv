# SPDX-License-Identifier: Apache-2.0

"""把一次完整 KV restore 表达成可滚动激活的 prefetch plan。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from vllm.core.block_reservation import BlockMapping, LogicalBlockKey


def _env(values: Mapping[str, str], name: str, _legacy_name: str,
         default: str) -> str:
    """Read a canonical GranuleKV setting.

    The unused legacy parameter is kept locally so the existing plan-building
    call sites remain easy to audit while no old namespace is consulted.
    """
    return values.get(name, default)


@dataclass(frozen=True)
class PrefetchBlockSelectorConfig:
    """把父 reservation 的 block 集合裁剪成可 profiling 的小粒度 I/O。

    默认 ``dense`` 保持现有 layer-prefetch 语义：每个 unit 都恢复父
    reservation 的全部 blocks。非 dense selector 默认仍是 profiling-only；
    只有显式的 ``consumer_enabled`` sparse 模式才允许真实 consumer 使用它，
    且不会把部分恢复发布成完整 prefix hash。
    """

    policy: str = "dense"
    block_count: int = 0
    stride: int = 1

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "PrefetchBlockSelectorConfig":
        values = os.environ if environ is None else environ
        policy = values.get("VLLM_GRANULEKV_PREFETCH_BLOCK_SELECTOR",
                           "dense").strip().lower()
        block_count = int(
            values.get("VLLM_GRANULEKV_PREFETCH_BLOCK_COUNT", "0"))
        stride = int(values.get("VLLM_GRANULEKV_PREFETCH_BLOCK_STRIDE", "1"))

        if policy in ("", "none", "all"):
            policy = "dense"
        if policy not in ("dense", "tail_n", "recent_n", "stride"):
            raise ValueError(
                "VLLM_GRANULEKV_PREFETCH_BLOCK_SELECTOR must be one of "
                "dense, tail_n, recent_n, stride")
        if policy in ("tail_n", "recent_n") and block_count <= 0:
            raise ValueError(
                "VLLM_GRANULEKV_PREFETCH_BLOCK_COUNT must be positive for "
                f"{policy}")
        if policy == "stride" and stride <= 0:
            raise ValueError(
                "VLLM_GRANULEKV_PREFETCH_BLOCK_STRIDE must be positive")
        return cls(policy=policy, block_count=block_count, stride=stride)

    @property
    def is_dense(self) -> bool:
        return self.policy == "dense"

    @property
    def profiling_only(self) -> bool:
        """非 dense selector 目前只用于 I/O profiling，不发布完整 prefix。"""
        return not self.is_dense

    def select(self, total_blocks: int) -> Optional[Tuple[int, ...]]:
        """返回 reservation-relative block indices；dense 返回 ``None``。

        这里故意只提供两个最小 selector：
        - ``tail_n/recent_n``：模拟 sparse attention 常见的近邻 KV 访问。
        - ``stride``：制造可重复的稀疏采样，用于观察随机/跨段 restore 开销。
        后续真实 sparse mask 也应只替换这个选择函数，而不改 GranuleKV 生命周期。
        """
        if total_blocks <= 0:
            raise ValueError("prefetch selector requires at least one block")
        if self.policy == "dense":
            return None
        if self.policy in ("tail_n", "recent_n"):
            count = min(self.block_count, total_blocks)
            return tuple(range(total_blocks - count, total_blocks))
        if self.policy == "stride":
            return tuple(range(0, total_blocks, self.stride))
        raise AssertionError(f"unreachable selector policy: {self.policy}")


@dataclass(frozen=True)
class SparseKVAccessPlan:
    """描述每一层实际会访问哪些逻辑 KV blocks。

    ``block_indices_by_layer[layer]`` 使用 prefix block table 中的逻辑下标，
    而不是 GranuleKV mapping 中的相对下标。这样同一份计划既能覆盖已经在 HBM
    命中的 block，也能覆盖需要从 SSD 恢复的 block；后端只负责把两者的
    交集转换成真实 I/O。

    ``None`` 表示该层使用完整 prefix，即 dense attention。未来真实 sparse
    attention 只需要为每层填入 block 下标，不需要修改 scheduler/GranuleKV 的
    stage、activate、poll 和 barrier 生命周期。
    """

    num_layers: int
    num_blocks: int
    block_indices_by_layer: Tuple[Optional[Tuple[int, ...]], ...]
    source: str = "dense"

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("sparse KV access plan requires layers")
        if self.num_blocks <= 0:
            raise ValueError("sparse KV access plan requires blocks")
        if len(self.block_indices_by_layer) != self.num_layers:
            raise ValueError(
                "one sparse KV block selection is required per layer")
        for indices in self.block_indices_by_layer:
            if indices is None:
                continue
            if not indices:
                raise ValueError("sparse KV layer selection must not be empty")
            if any(index < 0 or index >= self.num_blocks for index in indices):
                raise ValueError("sparse KV block index is outside prefix")
            if any(left >= right for left, right in zip(indices, indices[1:])):
                raise ValueError(
                    "sparse KV block indices must be strictly increasing")

    @classmethod
    def from_layer_selections(
        cls,
        *,
        num_blocks: int,
        block_indices_by_layer: Sequence[Sequence[int]],
        source: str = "dynamic",
    ) -> "SparseKVAccessPlan":
        """Normalize policy output into an immutable prefix access plan."""
        selections = tuple(
            tuple(sorted(set(int(index) for index in indices)))
            for indices in block_indices_by_layer)
        if not selections or any(not indices for indices in selections):
            raise ValueError("sparse layer selections must not be empty")
        if any(index < 0 or index >= num_blocks for indices in selections
               for index in indices):
            raise ValueError("sparse selection is outside the prefix")
        return cls(num_layers=len(selections),
                   num_blocks=num_blocks,
                   block_indices_by_layer=selections,
                   source=source)

    @classmethod
    def from_selector(
        cls,
        *,
        num_layers: int,
        num_blocks: int,
        selector: PrefetchBlockSelectorConfig,
    ) -> "SparseKVAccessPlan":
        """把现有环境变量 selector 转成逐层访问计划。

        当前 ``tail_n``/``stride`` 在所有层使用同一 block 集合，便于做可重复
        profiling。真实 sparse attention 后续可以直接构造本类，为不同层提供
        不同集合；下面的 PrefetchPlan 会自动按 layer window 取并集。
        """
        selected = selector.select(num_blocks)
        return cls(num_layers=num_layers,
                   num_blocks=num_blocks,
                   block_indices_by_layer=(selected, ) * num_layers,
                   source=selector.policy)

    @property
    def is_dense(self) -> bool:
        return all(indices is None
                   for indices in self.block_indices_by_layer)

    def blocks_for_layer(self,
                         layer_index: int) -> Optional[Tuple[int, ...]]:
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(f"layer is outside sparse KV plan: {layer_index}")
        return self.block_indices_by_layer[layer_index]

    def blocks_for_range(
        self,
        start_layer: int,
        end_layer: int,
    ) -> Optional[Tuple[int, ...]]:
        """返回一个 layer window 需要预取的 block 并集。

        一个 window 内只要有一层仍是 dense，就必须恢复完整 prefix；否则会
        让 dense consumer 看到未恢复地址。纯 sparse window 则只提交各层访问
        集合的并集，层内更精确的选择仍保留在本计划中供 consumer 查询。
        """
        if not 0 <= start_layer < end_layer <= self.num_layers:
            raise ValueError("invalid sparse KV layer range")
        selections = self.block_indices_by_layer[start_layer:end_layer]
        if any(indices is None for indices in selections):
            return None
        return tuple(sorted({index for indices in selections
                             for index in indices or ()}))


@dataclass(frozen=True)
class PrefetchUnit:
    """一个可独立激活和等待的细粒度 prefetch 单元。

    ``start_layer/end_layer`` 描述数据首次被消费的模型层范围；
    ``block_indices`` 描述本单元需要从父 reservation 中读取哪些 token
    blocks。二者正交，因此 layer prefetch 使用“一个层窗 + 全部 blocks”，
    sparse attention 后续可以使用“一个层窗 + 部分 blocks”，而无需改变
    stage/activate/poll/wait 四段生命周期。

    ``block_indices=None`` 是显式的 dense 语义，即选择父 reservation 的
    全部 blocks。索引均相对父 reservation，而不是 SSD/GPU 物理 block id，
    从而让 plan 保持后端无关，也不持有 allocator 生命周期。
    """

    index: int
    start_layer: int
    end_layer: int
    block_indices: Optional[Tuple[int, ...]] = None
    # 与 layer_range 等长；每一项是该层 attention 真正允许访问的 blocks。
    # block_indices 则是这些集合的并集，专供 GranuleKV 一次预取整个 window。
    consumer_blocks_by_layer: Optional[
        Tuple[Optional[Tuple[int, ...]], ...]] = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("prefetch unit index must be non-negative")
        if self.start_layer < 0 or self.end_layer <= self.start_layer:
            raise ValueError("prefetch unit layer range must be non-empty")
        if (self.consumer_blocks_by_layer is not None
                and len(self.consumer_blocks_by_layer) != self.num_layers):
            raise ValueError(
                "one consumer block selection is required per unit layer")
        if self.block_indices is None:
            return
        if not self.block_indices:
            raise ValueError("prefetch unit block selection must be non-empty")
        if any(index < 0 for index in self.block_indices):
            raise ValueError("prefetch block index must be non-negative")
        if any(left >= right for left, right in zip(
                self.block_indices, self.block_indices[1:])):
            raise ValueError(
                "prefetch block indices must be strictly increasing")

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer

    @property
    def layer_range(self) -> Tuple[int, int]:
        return self.start_layer, self.end_layer


@dataclass(frozen=True)
class PrefetchPlan:
    """一次 prefix restore 的不可变 prefetch plan。

    plan 只保存消费范围以及相对 block 选择，不持有物理 block、
    SequenceGroup 或 GranuleKV handle。scheduler 和 model runner 可以共享它，
    而不会把 allocator 生命周期耦合进策略模块。
    """

    plan_id: str
    units: Tuple[PrefetchUnit, ...]
    created_monotonic_ns: int
    block_selector: str = "dense"
    profiling_only: bool = False
    access_plan: Optional[SparseKVAccessPlan] = None
    # Default false preserves the old non-dense profiling-only behavior.
    consumer_enabled: bool = False
    # False preserves fixed-window union semantics.  True means units were
    # split at every change of the per-layer prefix selection.
    exact_grouped: bool = False
    # Human-readable mode used by logs and experiment aggregation.
    restore_mode: str = "window_union"

    @property
    def first_unit(self) -> PrefetchUnit:
        return self.units[0]

    def unit_for_layer(self, layer_index: int) -> PrefetchUnit:
        """返回消费 ``layer_index`` 的 unit，并尽早拒绝错误模型配置。"""
        for unit in self.units:
            if unit.start_layer <= layer_index < unit.end_layer:
                return unit
        raise ValueError(f"layer is outside prefetch plan: {layer_index}")

    def consumer_blocks_for_layer(
        self,
        layer_index: int,
    ) -> Optional[Tuple[int, ...]]:
        """返回 attention consumer 应使用的逻辑 block 集合。

        ``None`` 保持 dense attention 语义。非 ``None`` 时，consumer 必须只
        访问返回的 blocks，并在访问前通过 worker residency 校验。
        """
        if self.access_plan is None:
            unit = self.unit_for_layer(layer_index)
            if unit.consumer_blocks_by_layer is None:
                return unit.block_indices
            return unit.consumer_blocks_by_layer[layer_index -
                                                  unit.start_layer]
        return self.access_plan.blocks_for_layer(layer_index)


def select_prefetch_unit_blocks(
    unit: PrefetchUnit,
    block_mapping: BlockMapping,
    logical_blocks: Sequence[LogicalBlockKey],
) -> Tuple[BlockMapping, Tuple[LogicalBlockKey, ...]]:
    """把父 reservation 投影为一个 prefetch unit 的实际 I/O 数据集。

    mapping 与 logical key 必须始终一一对应，后者会在完成后更新 residency
    directory。选择动作集中在这里，可以避免 layer/sparse 策略分别复制
    scheduler 入队逻辑，也能在提交 GranuleKV 前统一拦截越界或错位的计划。
    """
    mapping = tuple(block_mapping)
    keys = tuple(logical_blocks)
    if len(mapping) != len(keys):
        raise ValueError(
            "block mapping and logical blocks must have the same length")

    # None 是当前 layer prefetch 的默认值：每个 layer window 都读取完整
    # prefix 中尚未驻留 HBM 的数据。这里返回原 tuple，不创建中间对象。
    if unit.block_indices is None:
        return mapping, keys

    # sparse plan 使用完整 prefix 的逻辑 block 下标；reservation mapping 只
    # 包含尚未在 HBM 的 SSD 扩展部分，因此必须按 LogicalBlockKey 求交集，
    # 不能再把 sparse 下标误当作 mapping 的数组下标。
    selected = frozenset(unit.block_indices)
    projected = tuple((pair, key) for pair, key in zip(mapping, keys)
                      if key.logical_index in selected)
    return (tuple(pair for pair, _ in projected),
            tuple(key for _, key in projected))


@dataclass(frozen=True)
class RollingPrefetchConfig:
    """worker-local rolling activation 配置，默认关闭。

    ``lead_units`` 表示模型消费当前 unit 时，至少再激活多少个未来 unit。
    ``target_slack_ms`` 为 0 时只使用固定 lead；大于 0 时，如果某次
    barrier 仍发生等待，runtime 会临时把 lead 增大 1，最多到
    ``max_lead_units``。这是最小的反馈策略，不在 scheduler 中预测 I/O。

    环境变量继续沿用已发布的 ``*_LEAD_WINDOWS`` 名称，只在读取边界转换为
    通用 unit 语义，避免破坏现有评测脚本和历史实验配置。
    """

    enabled: bool = False
    lead_units: int = 1
    max_lead_units: int = 1
    target_slack_ms: float = 0.0
    working_set_enabled: bool = False

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "RollingPrefetchConfig":
        values = os.environ if environ is None else environ
        enabled = bool(
            int(_env(values, "VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE",
                     "VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE", "0")))
        if not enabled:
            return cls()
        lead_units = int(_env(
            values, "VLLM_GRANULEKV_HIERARCHICAL_LEAD_WINDOWS",
            "VLLM_GRANULEKV_HIERARCHICAL_LEAD_WINDOWS", "1"))
        max_lead_units = int(_env(
            values, "VLLM_GRANULEKV_HIERARCHICAL_MAX_LEAD_WINDOWS",
            "VLLM_GRANULEKV_HIERARCHICAL_MAX_LEAD_WINDOWS",
            str(lead_units)))
        target_slack_ms = float(_env(
            values, "VLLM_GRANULEKV_HIERARCHICAL_TARGET_SLACK_MS",
            "VLLM_GRANULEKV_HIERARCHICAL_TARGET_SLACK_MS", "0"))
        working_set_enabled = bool(
            int(_env(values, "VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE",
                     "VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE", "0")))
        if lead_units < 0:
            raise ValueError(
                "VLLM_GRANULEKV_HIERARCHICAL_LEAD_WINDOWS must be non-negative")
        if max_lead_units < lead_units:
            raise ValueError(
                "VLLM_GRANULEKV_HIERARCHICAL_MAX_LEAD_WINDOWS must be >= "
                "VLLM_GRANULEKV_HIERARCHICAL_LEAD_WINDOWS")
        if target_slack_ms < 0:
            raise ValueError(
                "VLLM_GRANULEKV_HIERARCHICAL_TARGET_SLACK_MS must be non-negative")
        return cls(enabled=True,
                   lead_units=lead_units,
                   max_lead_units=max_lead_units,
                   target_slack_ms=target_slack_ms,
                   working_set_enabled=working_set_enabled)

    @property
    def initial_units(self) -> int:
        """首批激活 unit 数；之后由模型进度滚动激活。"""
        return max(1, self.lead_units)


def get_layer_working_set_regions(
        environ: Optional[Mapping[str, str]] = None) -> int:
    """计算 layer working-set 的真实 GPU region 数，关闭时返回 0。

    当前 window 消费期间还要允许 ``max_lead_units`` 个未来 window 读入，
    因此 region 数固定为 ``window_layers * (max_lead_units + 1)``。
    """
    values = os.environ if environ is None else environ
    if not bool(int(_env(values, "VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE",
                         "VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE", "0"))):
        return 0
    required_switches = (
        "VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE",
        "VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER",
        "VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE",
    )
    legacy_switches = {
        "VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE":
        "VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE",
        "VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER":
        "VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER",
        "VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE":
        "VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE",
    }
    if any(not bool(int(_env(values, name, legacy_switches[name], "0")))
           for name in required_switches):
        raise ValueError(
            "layer working set requires hierarchical I/O, layer barrier and "
            "rolling prefetch")
    num_layers = int(_env(values, "VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS",
                          "VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "0"))
    window_layers = int(
        _env(values, "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS",
             "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "0"))
    max_lead = int(_env(
        values, "VLLM_GRANULEKV_HIERARCHICAL_MAX_LEAD_WINDOWS",
        "VLLM_GRANULEKV_HIERARCHICAL_MAX_LEAD_WINDOWS",
        _env(values, "VLLM_GRANULEKV_HIERARCHICAL_LEAD_WINDOWS",
             "VLLM_GRANULEKV_HIERARCHICAL_LEAD_WINDOWS", "1")))
    if num_layers <= 0 or window_layers <= 0 or max_lead < 0:
        raise ValueError("invalid layer working-set configuration")
    regions = window_layers * (max_lead + 1)
    if regions > num_layers:
        raise ValueError("layer working set cannot exceed model layer count")
    return regions


@dataclass(frozen=True)
class HierarchicalIOConfig:
    """默认关闭的层级 restore 实验配置。"""

    enabled: bool = False
    num_layers: int = 0
    window_layers: int = 0
    rolling: RollingPrefetchConfig = RollingPrefetchConfig()
    block_selector: PrefetchBlockSelectorConfig = (
        PrefetchBlockSelectorConfig())
    consumer_enabled: bool = False
    exact_restore: bool = False

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "HierarchicalIOConfig":
        values = os.environ if environ is None else environ
        enabled = bool(
            int(_env(values, "VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE",
                     "VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "0")))
        if not enabled:
            return cls()

        # SchedulerConfig 不携带模型层数；这里显式使用 PP rank 的本地层数，
        # 避免从模型名称或全局层数猜测。单卡模型通常就是总 hidden layers。
        num_layers = int(_env(
            values, "VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS",
            "VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "0"))
        window_layers = int(_env(
            values, "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS",
            "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "0"))
        if num_layers <= 0:
            raise ValueError(
                "VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS must be positive")
        if window_layers <= 0:
            raise ValueError(
                "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS must be positive")
        return cls(enabled=True,
                   num_layers=num_layers,
                   window_layers=window_layers,
                   rolling=RollingPrefetchConfig.from_env(values),
                   block_selector=PrefetchBlockSelectorConfig.from_env(
                       values),
                   consumer_enabled=bool(
                       int(_env(values, "VLLM_GRANULEKV_SPARSE_CONSUMER_ENABLE",
                                "VLLM_GRANULEKV_SPARSE_CONSUMER_ENABLE", "0"))),
                   exact_restore=bool(
                       int(_env(
                           values,
                           "VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE",
                           "VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE",
                           "0"))))

    def build_plan(
        self,
        plan_id: str,
        num_blocks: Optional[int] = None,
        access_plan: Optional[SparseKVAccessPlan] = None,
    ) -> PrefetchPlan:
        if not self.enabled:
            raise RuntimeError("hierarchical I/O is disabled")
        return build_layer_restore_plan(plan_id=plan_id,
                                        num_layers=self.num_layers,
                                        window_layers=self.window_layers,
                                        block_selector=self.block_selector,
                                        num_blocks=num_blocks,
                                        access_plan=access_plan,
                                        consumer_enabled=self.consumer_enabled,
                                        exact_grouped=self.exact_restore)


def build_layer_restore_plan(
    *,
    plan_id: str,
    num_layers: int,
    window_layers: int,
    created_monotonic_ns: Optional[int] = None,
    block_selector: PrefetchBlockSelectorConfig = (
        PrefetchBlockSelectorConfig()),
    num_blocks: Optional[int] = None,
    access_plan: Optional[SparseKVAccessPlan] = None,
    consumer_enabled: bool = False,
    exact_grouped: bool = False,
) -> PrefetchPlan:
    """按模型执行顺序生成连续、无重叠、无空洞的 layer windows。"""
    if not plan_id:
        raise ValueError("plan_id must not be empty")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if window_layers <= 0:
        raise ValueError("window_layers must be positive")
    if access_plan is None:
        if block_selector.is_dense:
            # dense 默认路径不构造逐层 block tuple，保持原 layer prefetch 的
            # 对象数量和热路径不变。只有 sparse selector 才建立 access plan。
            block_indices_by_unit = None
        else:
            if num_blocks is None:
                raise ValueError(
                    "num_blocks is required for sparse prefetch selector")
            access_plan = SparseKVAccessPlan.from_selector(
                num_layers=num_layers,
                num_blocks=num_blocks,
                selector=block_selector,
            )
            block_indices_by_unit = access_plan
    else:
        if access_plan.num_layers != num_layers:
            raise ValueError("access plan layer count does not match model")
        if num_blocks is not None and access_plan.num_blocks != num_blocks:
            raise ValueError("access plan block count does not match prefix")
        block_indices_by_unit = access_plan

    if exact_grouped and access_plan is not None:
        # A unit is allowed to carry one exact prefix set only when every layer
        # in its range consumes that same set.  This keeps SSD mapping exact;
        # the per-layer consumer tuples still remain attached to the unit.
        exact_units = []
        start = 0
        while start < num_layers:
            selected = access_plan.blocks_for_layer(start)
            end = start + 1
            while (end < num_layers and end - start < window_layers
                   and access_plan.blocks_for_layer(end) == selected):
                end += 1
            exact_units.append(
                PrefetchUnit(
                    index=len(exact_units),
                    start_layer=start,
                    end_layer=end,
                    block_indices=selected,
                    consumer_blocks_by_layer=(
                        access_plan.block_indices_by_layer[start:end])))
            start = end
        units = tuple(exact_units)
    else:
        units = tuple(
            PrefetchUnit(
                index=index,
                start_layer=start,
                end_layer=min(start + window_layers, num_layers),
                block_indices=(
                    None if block_indices_by_unit is None else
                    block_indices_by_unit.blocks_for_range(
                        start, min(start + window_layers, num_layers))),
                consumer_blocks_by_layer=(
                    None if access_plan is None else
                    access_plan.block_indices_by_layer[
                        start:min(start + window_layers, num_layers)]))
            for index, start in enumerate(range(0, num_layers, window_layers)))
    profiling_only = (access_plan is not None and not access_plan.is_dense)
    return PrefetchPlan(
        plan_id=plan_id,
        units=units,
        created_monotonic_ns=(time.monotonic_ns()
                              if created_monotonic_ns is None else
                              created_monotonic_ns),
        block_selector=(access_plan.source
                        if access_plan is not None else block_selector.policy),
        profiling_only=profiling_only,
        access_plan=access_plan,
        consumer_enabled=consumer_enabled,
        exact_grouped=(exact_grouped and access_plan is not None),
        restore_mode=("exact_grouped"
                      if exact_grouped and access_plan is not None else
                      "window_union"),
    )
