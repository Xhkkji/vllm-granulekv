#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#include <algorithm>
#include <vector>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__global__ void score_pages_kernel(const scalar_t* __restrict__ query,
                                   const scalar_t* __restrict__ reps,
                                   float* __restrict__ scores, int num_heads,
                                   int num_pages, int kv_heads,
                                   int head_dim) {
  const int page = blockIdx.x;
  const int head = blockIdx.y;
  if (page >= num_pages || head >= num_heads) return;

  const int kv_head = head / (num_heads / kv_heads);
  float partial = 0.0f;
  for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float q = to_float(query[head * head_dim + dim]);
    const int envelope = q >= 0.0f ? 1 : 0;
    const int rep_index = (((page * 2 + envelope) * kv_heads + kv_head) *
                           head_dim + dim);
    partial += q * to_float(reps[rep_index]);
  }
  partial = warp_sum(partial);
  if (threadIdx.x == 0) scores[head * num_pages + page] = partial;
}

__device__ __forceinline__ bool comes_after(float left_score, int left_index,
                                             float right_score,
                                             int right_index) {
  return left_score < right_score ||
         (left_score == right_score && left_index > right_index);
}

__global__ void select_pages_kernel(const float* __restrict__ scores,
                                    int* __restrict__ output,
                                    int* __restrict__ counts, int num_heads,
                                    int num_pages, int num_blocks,
                                    int init_blocks, int local_blocks,
                                    int block_budget, int max_selected,
                                    int sort_count) {
  const int head = blockIdx.x;
  if (head >= num_heads) return;

  const int prefix_count = max(0, num_blocks - 1);
  const int init_end = min(init_blocks, prefix_count);
  const int local_start = max(init_end, prefix_count - local_blocks);
  const int candidate_count = max(0, local_start - init_end);
  const int dynamic_count = min(block_budget, candidate_count);
  extern __shared__ unsigned char shared_memory[];
  float* sorted_scores = reinterpret_cast<float*>(shared_memory);
  int* sorted_indices = reinterpret_cast<int*>(
      sorted_scores + sort_count);

  for (int index = threadIdx.x; index < sort_count; index += blockDim.x) {
    if (index < candidate_count) {
      const int logical = init_end + index;
      sorted_scores[index] = scores[head * num_pages + logical];
      sorted_indices[index] = logical;
    } else {
      sorted_scores[index] = -FLT_MAX;
      sorted_indices[index] = -1;
    }
  }
  __syncthreads();

  // Sort candidate pages in parallel.  The final order is descending score
  // with logical-index tie breaking, matching the deterministic selector.
  for (int k = 2; k <= sort_count; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int index = threadIdx.x; index < sort_count;
           index += blockDim.x) {
        const int partner = index ^ j;
        if (partner <= index || partner >= sort_count) continue;
        const bool ascending = (index & k) == 0;
        const bool swap_values =
            comes_after(sorted_scores[index], sorted_indices[index],
                        sorted_scores[partner], sorted_indices[partner]);
        if (swap_values == ascending) {
          const float score = sorted_scores[index];
          sorted_scores[index] = sorted_scores[partner];
          sorted_scores[partner] = score;
          const int logical = sorted_indices[index];
          sorted_indices[index] = sorted_indices[partner];
          sorted_indices[partner] = logical;
        }
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    int selected_count = 0;
    for (int logical = 0; logical < num_blocks; ++logical) {
      bool keep = logical < init_end || logical >= local_start;
      if (!keep && logical < prefix_count) {
        for (int rank = 0; rank < dynamic_count; ++rank) {
          keep |= sorted_indices[rank] == logical;
        }
      }
      if (keep) {
        output[head * max_selected + selected_count] = logical;
        ++selected_count;
      }
    }
    counts[head] = selected_count;
  }
}

template <typename scalar_t, typename block_t>
__global__ void sparse_decode_kernel(
    scalar_t* __restrict__ output, const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const block_t* __restrict__ block_table,
    const int* __restrict__ logical_indices, const int* __restrict__ counts,
    int table_stride, int index_stride, int count_stride, int num_heads,
    int max_selected, int head_dim, int kv_heads, int key_dim_chunks,
    int key_pack, int block_size, int sequence_length, float scale,
    int64_t key_stride_block, int64_t key_stride_head,
    int64_t key_stride_dim_chunk, int64_t key_stride_token,
    int64_t key_stride_pack, int64_t value_stride_block,
    int64_t value_stride_head, int64_t value_stride_dim,
    int64_t value_stride_token) {
  const int head = blockIdx.x;
  const int lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) return;

  const int group_size = num_heads / kv_heads;
  const int kv_head = head / group_size;
  const int selected_count = counts[head * count_stride];
  float accum[8] = {0.0f};
  const int local_dims = (head_dim + 31) / 32;
  float max_logit = -FLT_MAX;
  float denominator = 0.0f;

  for (int selected = 0; selected < selected_count; ++selected) {
    const int logical = logical_indices[head * index_stride + selected];
    if (logical < 0) continue;
    const int num_blocks = (sequence_length + block_size - 1) / block_size;
    if (logical >= num_blocks) continue;
    const int physical = static_cast<int>(
        block_table[logical * table_stride]);
    const int token_count = logical == num_blocks - 1
                                ? sequence_length - logical * block_size
                                : block_size;

    for (int token = 0; token < token_count; ++token) {
      float dot = 0.0f;
      for (int local = 0; local < local_dims; ++local) {
        const int dim = lane + local * 32;
        if (dim >= head_dim) continue;
        const int dim_chunk = dim / key_pack;
        const int pack_offset = dim % key_pack;
        const int64_t key_index =
            static_cast<int64_t>(physical) * key_stride_block +
            static_cast<int64_t>(kv_head) * key_stride_head +
            static_cast<int64_t>(dim_chunk) * key_stride_dim_chunk +
            static_cast<int64_t>(token) * key_stride_token + pack_offset *
                key_stride_pack;
        dot += to_float(query[head * head_dim + dim]) *
               to_float(key_cache[key_index]);
      }
      dot = warp_sum(dot) * scale;
      dot = __shfl_sync(0xffffffff, dot, 0);

      const float new_max = max(max_logit, dot);
      const float old_weight = expf(max_logit - new_max);
      const float token_weight = expf(dot - new_max);
      for (int local = 0; local < local_dims; ++local) {
        const int dim = lane + local * 32;
        if (dim >= head_dim) continue;
        const int64_t value_index =
            static_cast<int64_t>(physical) * value_stride_block +
            static_cast<int64_t>(kv_head) * value_stride_head +
            static_cast<int64_t>(dim) * value_stride_dim +
            static_cast<int64_t>(token) * value_stride_token;
        accum[local] = accum[local] * old_weight +
                       token_weight * to_float(value_cache[value_index]);
      }
      denominator = denominator * old_weight + token_weight;
      max_logit = new_max;
    }
  }

  for (int local = 0; local < local_dims; ++local) {
    const int dim = lane + local * 32;
    if (dim < head_dim) {
      output[head * head_dim + dim] =
          static_cast<scalar_t>(accum[local] / denominator);
    }
  }
}

void check_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

}  // namespace

void score_pages(torch::Tensor query, torch::Tensor representatives,
                 torch::Tensor scores) {
  check_cuda(query, "query");
  check_cuda(representatives, "representatives");
  TORCH_CHECK(query.dim() == 2, "query must have shape [heads, dim]");
  TORCH_CHECK(representatives.dim() == 4 && representatives.size(1) == 2,
              "representatives must have shape [pages, 2, kv_heads, dim]");
  TORCH_CHECK(query.scalar_type() == representatives.scalar_type(),
              "query and representatives must have the same dtype");
  TORCH_CHECK(query.size(0) % representatives.size(2) == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK(query.size(1) == representatives.size(3),
              "query and representatives have different head sizes");

  query = query.contiguous();
  representatives = representatives.contiguous();
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat &&
                  scores.sizes() ==
                      torch::IntArrayRef({query.size(0), representatives.size(0)}),
              "scores must have shape [heads, pages] and float CUDA dtype");
  const dim3 grid(representatives.size(0), query.size(0), 1);
  const auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      query.scalar_type(), "solidattention_score_pages", [&] {
        score_pages_kernel<scalar_t><<<grid, 32, 0, stream>>>(
            query.data_ptr<scalar_t>(), representatives.data_ptr<scalar_t>(),
            scores.data_ptr<float>(), query.size(0), representatives.size(0),
            representatives.size(2), query.size(1));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void select_pages(torch::Tensor scores, torch::Tensor indices,
                  torch::Tensor counts, int64_t num_blocks, int64_t init_blocks,
                                        int64_t local_blocks,
                                        int64_t block_budget) {
  check_cuda(scores, "scores");
  TORCH_CHECK(scores.dim() == 2 && scores.scalar_type() == torch::kFloat,
              "scores must be a CUDA float matrix");
  TORCH_CHECK(num_blocks > 0 && block_budget > 0,
              "num_blocks and block_budget must be positive");
  const int prefix_count = std::max<int64_t>(0, num_blocks - 1);
  const int init_end = std::min<int64_t>(init_blocks, prefix_count);
  const int local_start = std::max<int64_t>(init_end,
                                            prefix_count - local_blocks);
  const int dynamic_count = std::min<int64_t>(
      block_budget, std::max(0, local_start - init_end));
  const int suffix_count = static_cast<int>(num_blocks - prefix_count);
  const int max_selected = init_end + dynamic_count +
                           (prefix_count - local_start) + suffix_count;
  TORCH_CHECK(max_selected <= 4096,
              "SolidAttention runtime selection supports at most 4096 pages");
  TORCH_CHECK(scores.size(1) >= prefix_count,
              "scores do not cover the immutable prefix");
  TORCH_CHECK(indices.is_cuda() && counts.is_cuda() &&
                  indices.scalar_type() == torch::kInt &&
                  counts.scalar_type() == torch::kInt &&
                  indices.sizes() == torch::IntArrayRef(
                      {scores.size(0), max_selected}) &&
                  counts.sizes() == torch::IntArrayRef({scores.size(0)}),
              "invalid selection output buffers");
  const auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
  int sort_count = 1;
  while (sort_count < std::max(1, local_start - init_end)) {
    sort_count <<= 1;
  }
  const size_t shared_bytes =
      static_cast<size_t>(sort_count) * (sizeof(float) + sizeof(int));
  select_pages_kernel<<<scores.size(0), 256, shared_bytes, stream>>>(
      scores.data_ptr<float>(), indices.data_ptr<int>(), counts.data_ptr<int>(),
      scores.size(0), scores.size(1), num_blocks, init_end, local_blocks,
      block_budget, max_selected, sort_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename block_t>
void launch_decode(torch::Tensor output, torch::Tensor query,
                   torch::Tensor key_cache, torch::Tensor value_cache,
                   torch::Tensor block_table, torch::Tensor logical_indices,
                   torch::Tensor counts, int sequence_length, int block_size,
                   float scale, int num_kv_heads) {
  const auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
  sparse_decode_kernel<scalar_t, block_t><<<query.size(1), 32, 0, stream>>>(
      output.data_ptr<scalar_t>(), query.data_ptr<scalar_t>(),
      key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
      block_table.data_ptr<block_t>(), logical_indices.data_ptr<int>(),
      counts.data_ptr<int>(), block_table.stride(1), logical_indices.stride(0),
      counts.stride(0), query.size(1), logical_indices.size(1), query.size(2),
      num_kv_heads, key_cache.size(2), key_cache.size(4), block_size,
      sequence_length, scale, key_cache.stride(0), key_cache.stride(1),
      key_cache.stride(2), key_cache.stride(3), key_cache.stride(4),
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
      value_cache.stride(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void decode(torch::Tensor output, torch::Tensor query, torch::Tensor key_cache,
            torch::Tensor value_cache, torch::Tensor block_table,
            torch::Tensor logical_indices, torch::Tensor counts,
            int64_t sequence_length, int64_t block_size, double scale,
            int64_t num_kv_heads) {
  check_cuda(output, "output");
  check_cuda(query, "query");
  check_cuda(key_cache, "key_cache");
  check_cuda(value_cache, "value_cache");
  check_cuda(block_table, "block_table");
  check_cuda(logical_indices, "logical_indices");
  check_cuda(counts, "counts");
  TORCH_CHECK(query.dim() == 3 && output.sizes() == query.sizes(),
              "query/output must have shape [1, heads, dim]");
  TORCH_CHECK(query.size(0) == 1 && query.size(1) > 0,
              "runtime decode requires batch one");
  TORCH_CHECK(key_cache.dim() == 5 && value_cache.dim() == 4,
              "unsupported vLLM KV cache layout");
  TORCH_CHECK(key_cache.scalar_type() == torch::kFloat16 &&
                  value_cache.scalar_type() == torch::kFloat16 &&
                  query.scalar_type() == torch::kFloat16,
              "runtime decode currently supports FP16 only");
  TORCH_CHECK(output.scalar_type() == query.scalar_type(),
              "output and query must have the same dtype");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == 1,
              "block_table must have shape [1, blocks]");
  TORCH_CHECK(logical_indices.dim() == 2 && counts.dim() == 1 &&
                  logical_indices.size(0) == query.size(1) &&
                  counts.size(0) == query.size(1) &&
                  logical_indices.scalar_type() == torch::kInt &&
                  counts.scalar_type() == torch::kInt,
              "invalid per-head selection tensors");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt ||
                  block_table.scalar_type() == torch::kLong,
              "block_table must use int32 or int64");
  TORCH_CHECK(query.is_contiguous() && output.is_contiguous() &&
                  key_cache.is_contiguous() && value_cache.is_contiguous() &&
                  block_table.is_contiguous() &&
                  logical_indices.is_contiguous() && counts.is_contiguous(),
              "runtime decode inputs must be contiguous");
  TORCH_CHECK(num_kv_heads > 0 && query.size(1) % num_kv_heads == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK(key_cache.size(1) == num_kv_heads &&
                  value_cache.size(1) == num_kv_heads &&
                  key_cache.size(3) == block_size &&
                  value_cache.size(3) == block_size,
              "KV cache dimensions do not match runtime arguments");
  const int head_dim = query.size(2);
  TORCH_CHECK(value_cache.size(2) == head_dim &&
                  key_cache.size(2) * key_cache.size(4) == head_dim,
              "KV cache head dimensions do not match query");
  const int num_blocks = (sequence_length + block_size - 1) / block_size;
  TORCH_CHECK(block_table.size(1) >= num_blocks,
              "block table does not cover sequence_length");

  AT_DISPATCH_INTEGRAL_TYPES(
      block_table.scalar_type(), "solidattention_decode_block_table", [&] {
        launch_decode<at::Half, scalar_t>(
            output, query, key_cache, value_cache, block_table,
            logical_indices, counts, sequence_length, block_size,
            static_cast<float>(scale), static_cast<int>(num_kv_heads));
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("score_pages", &score_pages, "SolidAttention page scores");
  module.def("select_pages", &select_pages, "SolidAttention per-head pages");
  module.def("decode", &decode, "SolidAttention sparse decode");
}
