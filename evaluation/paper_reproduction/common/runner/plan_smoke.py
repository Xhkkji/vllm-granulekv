"""Build and validate one strategy plan without starting a model server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..plans import ImmediatePrefetcher, LayerWisePrefetcher, compose_plan
from ..schedulers import DiskHRRNScheduler, FCFSScheduler
from vllm.granulekv.sparse_policy.selectors import DenseSelector
from ...bidaw.adapter.bidaw import BidawScheduler
from ...hisparse.adapter.hisparse import HiSparseSelector
from ...quest.adapter.quest import QuestSelector
from vllm.granulekv.sparse_policy.solidattention import SolidAttentionSelector


def build_plan(name: str, layers: int, blocks: int, window: int,
               prefetcher_name: str = "layer_wise"):
    if name == "dense":
        selector = DenseSelector()
    elif name == "quest":
        selector = QuestSelector(block_budget=max(1, blocks // 4))
    elif name == "hisparse":
        selector = HiSparseSelector(block_budget=max(1, blocks // 4))
    elif name == "solidattention":
        selector = SolidAttentionSelector(block_budget=max(1, blocks // 4))
    elif name == "bidaw":
        selector = DenseSelector()
    elif name == "granulekv_joint":
        selector = QuestSelector(block_budget=max(1, blocks // 4))
    else:
        raise ValueError(f"unknown strategy: {name}")
    scheduler = (BidawScheduler() if name == "bidaw" else
                 DiskHRRNScheduler() if name == "granulekv_joint" else
                 FCFSScheduler())
    prefetcher = (ImmediatePrefetcher()
                  if prefetcher_name == "on_demand" else
                  LayerWisePrefetcher(window))
    return compose_plan(
        plan_id=f"paper-reproduction-{name}",
        selector=selector,
        prefetcher=prefetcher,
        scheduler=scheduler,
        num_layers=layers,
        num_blocks=blocks,
        metadata={"runner": "plan_smoke", "execution": "plan_only"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="dense",
                        choices=("dense", "quest", "hisparse",
                                 "solidattention", "bidaw",
                                 "granulekv_joint"))
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--prefetcher", choices=("on_demand", "layer_wise"),
                        default="layer_wise")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_plan(args.strategy, args.layers, args.blocks, args.window,
                      args.prefetcher)
    payload = json.dumps(plan.as_dict(), indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
