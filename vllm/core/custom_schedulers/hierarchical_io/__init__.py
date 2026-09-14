# SPDX-License-Identifier: Apache-2.0

"""GranuleKV 层级 I/O 的独立控制面组件。"""

from .barrier import (HierarchicalLayerBarrierConfig, activate_layer_barrier,
                      activate_sparse_kv_blocks,
                      get_active_sparse_kv_blocks, release_local_layer,
                      wait_for_local_layer, get_active_layer_request_ids,
                      get_active_layer_sequence_lengths)
from .lifecycle import (HierarchicalRestoreController,
                        HierarchicalRestoreProgress)
from .plan import (HierarchicalIOConfig, PrefetchBlockSelectorConfig,
                   PrefetchPlan, PrefetchUnit, RollingPrefetchConfig,
                            SparseKVAccessPlan, build_layer_restore_plan,
                   get_layer_working_set_regions,
                   select_prefetch_unit_blocks)
from .residency import PrefetchResidencyDirectory
from .runtime import PrefetchRuntimeTrace, RollingPrefetchRuntime
from .sparse_policy import (SparseKVPlanFeedback, SparseKVPolicy,
                            SparseKVPolicyRuntime,
                            bind_sparse_page_index_key,
                            build_sparse_restore_plan_feedback,
                            configure_sparse_kv_policy,
                            discard_sparse_restore_context,
                            get_sparse_kv_policy, load_sparse_kv_policy,
                            observe_sparse_query, select_sparse_blocks)
from .sparse_policy import (register_sparse_page_representatives,
                            register_sparse_restore_context)

__all__ = [
    "HierarchicalIOConfig",
    "HierarchicalLayerBarrierConfig",
    "HierarchicalRestoreController",
    "HierarchicalRestoreProgress",
    "PrefetchBlockSelectorConfig",
    "PrefetchPlan",
    "PrefetchResidencyDirectory",
    "PrefetchUnit",
    "RollingPrefetchConfig",
    "RollingPrefetchRuntime",
    "SparseKVAccessPlan",
    "PrefetchRuntimeTrace",
    "activate_layer_barrier",
    "activate_sparse_kv_blocks",
    "build_layer_restore_plan",
    "get_active_sparse_kv_blocks",
    "get_active_layer_request_ids",
    "get_active_layer_sequence_lengths",
    "get_layer_working_set_regions",
    "release_local_layer",
    "select_prefetch_unit_blocks",
    "wait_for_local_layer",
    "SparseKVPolicy",
    "SparseKVPlanFeedback",
    "SparseKVPolicyRuntime",
    "configure_sparse_kv_policy",
    "bind_sparse_page_index_key",
    "get_sparse_kv_policy",
    "load_sparse_kv_policy",
    "observe_sparse_query",
    "select_sparse_blocks",
    "build_sparse_restore_plan_feedback",
    "discard_sparse_restore_context",
    "register_sparse_page_representatives",
    "register_sparse_restore_context",
]
