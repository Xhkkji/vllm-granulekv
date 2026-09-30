# SPDX-License-Identifier: Apache-2.0
"""vLLM KV allocation metadata used by the GranuleKV transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


DTYPE_NAMES = {
    torch.uint8: "uint8",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for size in reversed(shape):
        result.append(stride)
        stride *= int(size)
    return tuple(reversed(result))


@dataclass(frozen=True)
class GranuleKVLayout:
    tensor_shape: tuple[int, ...]
    tensor_strides: tuple[int, ...]
    dtype_name: str
    element_size: int
    region_bytes: int
    fragment_bytes: int
    num_layers: int
    num_gpu_regions: int
    num_gpu_blocks: int
    num_storage_blocks: int
    device_index: int
    num_host_blocks: int = 0
    num_host_regions: int = 0
    host_region_bytes: int = 0
    host_kv_stride_elems: int = 0
    host_block_stride_elems: int = 0

    @classmethod
    def from_allocation(
        cls,
        *,
        allocation_shape: Sequence[int],
        stride_order: Sequence[int],
        dtype: torch.dtype,
        num_layers: int,
        num_gpu_regions: int,
        num_gpu_blocks: int,
        num_storage_blocks: int,
        device_index: int,
        num_host_blocks: int = 0,
        num_host_regions: int = 0,
    ) -> "GranuleKVLayout":
        shape = tuple(int(value) for value in allocation_shape)
        order = tuple(int(value) for value in stride_order)
        if sorted(order) != list(range(len(shape))):
            raise ValueError("invalid KV cache stride order")
        if dtype not in DTYPE_NAMES:
            raise ValueError(f"unsupported GranuleKV dtype: {dtype}")
        if num_storage_blocks <= 0:
            raise ValueError("GranuleKV requires positive storage blocks")
        if not 0 < num_gpu_regions <= num_layers:
            raise ValueError("invalid GranuleKV GPU layer-region count")

        allocation_strides = contiguous_strides(shape)
        tensor_shape = tuple(shape[index] for index in order)
        tensor_strides = tuple(allocation_strides[index] for index in order)
        if len(tensor_shape) != 3 or tensor_shape[0] != 2:
            raise ValueError(
                f"GranuleKV requires V0 [2,N,E] KV layout, got {tensor_shape}")
        if tensor_strides[2] != 1:
            raise ValueError("GranuleKV fragment must be contiguous")

        element_size = int(torch.empty((), dtype=dtype).element_size())
        storage_elements = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(tensor_shape, tensor_strides))
        region_bytes = storage_elements * element_size
        fragment_bytes = tensor_strides[1] * element_size
        if fragment_bytes % 4096 != 0:
            raise ValueError("GranuleKV fragment must be 4KB aligned")
        host_region_bytes = 0
        host_kv_stride_elems = 0
        host_block_stride_elems = 0
        if num_host_blocks:
            if num_host_blocks <= 0 or num_host_regions <= 0:
                raise ValueError("host blocks and regions must be positive together")
            host_block_stride_elems = int(tensor_strides[1])
            host_kv_stride_elems = int(num_host_blocks * host_block_stride_elems)
            host_shape = list(tensor_shape)
            host_shape[1] = int(num_host_blocks)
            host_strides = (host_kv_stride_elems, host_block_stride_elems,
                            int(tensor_strides[2]))
            host_storage_elements = 1 + sum(
                (size - 1) * stride
                for size, stride in zip(host_shape, host_strides))
            host_region_bytes = host_storage_elements * element_size
        elif num_host_regions:
            raise ValueError("host regions require host blocks")
        return cls(
            tensor_shape=tensor_shape,
            tensor_strides=tensor_strides,
            dtype_name=DTYPE_NAMES[dtype],
            element_size=element_size,
            region_bytes=region_bytes,
            fragment_bytes=fragment_bytes,
            num_layers=int(num_layers),
            num_gpu_regions=int(num_gpu_regions),
            num_gpu_blocks=int(num_gpu_blocks),
            num_storage_blocks=int(num_storage_blocks),
            device_index=int(device_index),
            num_host_blocks=int(num_host_blocks),
            num_host_regions=int(num_host_regions),
            host_region_bytes=int(host_region_bytes),
            host_kv_stride_elems=int(host_kv_stride_elems),
            host_block_stride_elems=int(host_block_stride_elems),
        )

    @property
    def host_tensor_shape(self) -> tuple[int, ...]:
        if self.num_host_blocks <= 0:
            return ()
        shape = list(self.tensor_shape)
        shape[1] = self.num_host_blocks
        return tuple(shape)

    @property
    def host_tensor_strides(self) -> tuple[int, ...]:
        if self.num_host_blocks <= 0:
            return ()
        return (self.host_kv_stride_elems, self.host_block_stride_elems,
                self.tensor_strides[2])

    def allocation_payload(self, *, client_pid: int) -> dict[str, Any]:
        return {
            "client_pid": int(client_pid),
            "device_index": self.device_index,
            "num_layers": self.num_layers,
            "num_gpu_regions": self.num_gpu_regions,
            "num_gpu_blocks": self.num_gpu_blocks,
            "num_storage_blocks": self.num_storage_blocks,
            "region_bytes": self.region_bytes,
            "element_size": self.element_size,
            "fragment_bytes": self.fragment_bytes,
            "kv_stride_elems": self.tensor_strides[0],
            "block_stride_elems": self.tensor_strides[1],
            "logical_block_bytes": self.num_layers * 2 * self.fragment_bytes,
            "num_host_blocks": self.num_host_blocks,
            "num_host_regions": self.num_host_regions,
            "host_region_bytes": self.host_region_bytes,
            "host_kv_stride_elems": self.host_kv_stride_elems,
            "host_block_stride_elems": self.host_block_stride_elems,
            "tensor_shape": list(self.tensor_shape),
            "tensor_strides": list(self.tensor_strides),
            "dtype": self.dtype_name,
        }

    def validate_manifest(self, manifest: dict[str, Any]) -> None:
        if (tuple(manifest["tensor_shape"]) != self.tensor_shape
                or tuple(manifest["tensor_strides"]) != self.tensor_strides
                or manifest["dtype"] != self.dtype_name):
            raise RuntimeError("GranuleKV allocation manifest layout mismatch")
        expected_regions = self.num_gpu_regions + self.num_host_regions
        if len(manifest["regions"]) != expected_regions:
            raise RuntimeError("GranuleKV allocation region count mismatch")
        if self.num_host_blocks:
            if len(manifest.get("gpu_regions", ())) != self.num_gpu_regions:
                raise RuntimeError("GranuleKV GPU region count mismatch")
            if len(manifest.get("host_regions", ())) != self.num_host_regions:
                raise RuntimeError("GranuleKV host region count mismatch")
