# SPDX-License-Identifier: Apache-2.0

"""固定总长度、扫描 prefix hit rate 的 Figure 13 workload 定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


DEFAULT_PREFIX_TOKENS = (16384, 24576, 28672, 30720, 31744, 32768)


@dataclass(frozen=True)
class Figure13Point:
    """一个固定总 prompt 中的 prefix/suffix 分割点。"""

    prefix_tokens: int
    suffix_tokens: int
    total_tokens: int

    @property
    def hit_rate(self) -> float:
        return self.prefix_tokens / self.total_tokens

    @property
    def label(self) -> str:
        return f"p{self.prefix_tokens}-s{self.suffix_tokens}"


@dataclass(frozen=True)
class Figure13SweepConfig:
    """经过 block 对齐检查的不可变 sweep 配置。

    这个对象不依赖 tokenizer、Scheduler 或 GranuleKV。评测 runner 和单元测试
    使用同一份配置生成实验点，避免 shell、Python runner 和结果聚合器各自
    维护一套 prefix 比例。
    """

    total_tokens: int
    block_size: int
    points: Tuple[Figure13Point, ...]

    @classmethod
    def create(
        cls,
        *,
        total_tokens: int = 32768,
        prefix_tokens: Sequence[int] = DEFAULT_PREFIX_TOKENS,
        block_size: int = 16,
    ) -> "Figure13SweepConfig":
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        prefixes = tuple(int(value) for value in prefix_tokens)
        if not prefixes:
            raise ValueError("prefix_tokens must not be empty")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("prefix_tokens must not contain duplicates")
        if tuple(sorted(prefixes)) != prefixes:
            raise ValueError("prefix_tokens must be strictly increasing")
        for prefix in prefixes:
            if not 0 <= prefix <= total_tokens:
                raise ValueError("prefix token count is outside total prompt")
            if prefix % block_size != 0:
                raise ValueError("prefix_tokens must align to block_size")

        return cls(
            total_tokens=total_tokens,
            block_size=block_size,
            points=tuple(
                Figure13Point(prefix_tokens=prefix,
                              suffix_tokens=total_tokens - prefix,
                              total_tokens=total_tokens)
                for prefix in prefixes),
        )


def parse_prefix_tokens(value: str) -> Tuple[int, ...]:
    """解析逗号分隔的 token 数；范围和对齐由 sweep config 统一验证。"""
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",")
                       if item.strip())
    except ValueError as exc:
        raise ValueError("prefix token list must contain integers") from exc
    if not parsed:
        raise ValueError("prefix token list must not be empty")
    return parsed


def build_reuse_prompt_tokens(
    base_tokens: Sequence[int],
    replacement_tokens: Sequence[int],
    point: Figure13Point,
) -> Tuple[int, ...]:
    """构造“命中 P tokens、重新计算 S tokens”的精确 token prompt。

    runner 直接把 token IDs 交给 vLLM，不经历 decode/re-encode，因此 block
    hash 的共享边界严格等于 ``point.prefix_tokens``。replacement 从 prefix
    边界开始使用；100% hit 点自然返回完整 base prompt。
    """
    if len(base_tokens) < point.total_tokens:
        raise ValueError("base token source is shorter than total prompt")
    if len(replacement_tokens) < point.total_tokens:
        raise ValueError("replacement token source is shorter than total prompt")
    prefix = tuple(base_tokens[:point.prefix_tokens])
    suffix = tuple(
        replacement_tokens[point.prefix_tokens:point.total_tokens])
    prompt = prefix + suffix
    if len(prompt) != point.total_tokens:
        raise RuntimeError("Figure 13 prompt length invariant was violated")
    return prompt
