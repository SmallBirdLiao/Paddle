/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#ifdef PADDLE_WITH_HETERPS
//#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include <queue>
#include <cooperative_groups.h>

namespace paddle {
namespace framework {

#define LXCH_PADDLE_ENFORCE_GPU_SUCCESS(COND)                  \
  do {                                                         \
    auto __cond__ = (COND);                                    \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);           \
    constexpr auto __success_type__ =                          \
        ::paddle::platform::details::ExternalApiType<          \
            __CUDA_STATUS_TYPE__>::kSuccess;                   \
    if (UNLIKELY(__cond__ != __success_type__)) {              \
      abort();                                                 \
    }                                                          \
  } while (0)

template <typename T>
__global__ void fill_idx(T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    idx[i] = i;
  }
}

__global__ void fill_key(uint64_t* keys, revert_info* revert, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    keys[i] = revert[i].key;
  }
}

__global__ void fill_revert_idx(uint64_t* keys, uint32_t* key_size, uint32_t* ken_offset, revert_info* result, uint32_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    result[i].offset = ken_offset[i];
    result[i].len = key_size[i];
    result[i].key = keys[i];
  }
}


template <typename T>
void show_tensor(T* input, size_t len, gpuStream_t stream, std::string name) {
  T tmp[len];  // NOLINT
  cudaMemcpyAsync(&tmp, input, sizeof(T) * len, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::cout << name;
  for (int i = 0; i < len; ++i) {
    std::cout << ":" << tmp[i];
  }
  std::cout << std::endl;
}

template <typename T>
__global__ void calc_shard_offset(T* idx, T* left, T* right, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len - 1) {
    if (idx[i] != idx[i + 1]) {
      right[idx[i]] = i;
      left[idx[i + 1]] = i + 1;
    }
  }
  if (i == 0) {
    left[idx[i]] = i;
  }
  if (i == (len - 1)) {
    right[idx[i]] = i;
  }
}

/*
__global__ void compute_normal_size(uint32_t* uniq_size, int* normal_size, uint32_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    uint32_t cur_size = uniq_size[i];
    if (i == 0) {
      if (cur_size < 8) {
        normal_size[0] = 1;
      } else if (cur_size < 32) {
        normal_size[1] = 1;
      } else {
        normal_size[2] = 1;
      }
    } else {
      uint32_t pre_next = uniq_size[i - 1];
      if (pre_next < 8 && cur_size >= 8) {
        if (cur_size < 32) {
          normal_size[1] = i + 1;
        } else {
          normal_size[2] = i + 1;
        }
      } else if (pre_next < 32 && cur_size >= 32) {
        normal_size[2] = i + 1;
      }
    }
  }
}
*/

__global__ void compute_normal_size(uint32_t* uniq_size, int* normal_size, uint32_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    uint32_t cur_size = uniq_size[i];
    if (i == 0) {
      if (cur_size < 4) {
        normal_size[0] = 1;
      } else if (cur_size < 16) {
        normal_size[1] = 1;
      } else if (cur_size < 64) {
        normal_size[2] = 1;
      } else if (cur_size < 256) {
        normal_size[3] = 1;
      } else {
        normal_size[4] = 1;
      }
    } else {
      uint32_t pre_next = uniq_size[i - 1];
      if (pre_next < 4) {
        if (cur_size >= 256) {
          normal_size[4] = i + 1;
        } else if (cur_size >= 64) {
          normal_size[3] = i + 1;
        } else if (cur_size >= 16) {
          normal_size[2] = i + 1;
        } else if (cur_size >= 4) {
          normal_size[1] = i + 1;
        }
      } else if (pre_next < 16) {
        if (cur_size >= 256) {
          normal_size[4] = i + 1;
        } else if (cur_size >= 64) {
          normal_size[3] = i + 1;
        } else if (cur_size >= 16) {
          normal_size[2] = i + 1;
        }
      } else if (pre_next < 64) {
        if (cur_size >= 256) {
          normal_size[4] = i + 1;
        } else if (cur_size >= 64) {
          normal_size[3] = i + 1;
        }
      } else if (pre_next < 256) {
        if (cur_size >= 256) {
          normal_size[4] = i + 1;
        }
      }
    }
  }
}


template <typename KeyType, typename T>
__global__ void calc_shard_index(KeyType* d_keys, size_t len, T* shard_index,
                                 int total_gpu) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i] % total_gpu;
  }
}

template <typename T>
__global__ void calc_shard_index(revert_info* d_keys, size_t len, T* shard_index,
                                 int total_gpu) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i].key % total_gpu;
  }
}

template <typename KeyType, typename T>
__global__ void fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys, T* idx,
                               size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
  }
}

__global__ void fill_reserve_shard_inx(uint32_t* d_shard_idx, uint32_t* d_idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_idx[d_idx[i]] = i;
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                 GradType* d_shard_grads, GradType* d_grads,
                                 T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    d_shard_grads[i] = d_grads[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void dy_mf_fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                       GradType* d_shard_grads,
                                       GradType* d_grads, T* idx, size_t len,
                                       size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    *(GradType*)((char*)d_shard_grads + i * grad_value_size) =
        *(GradType*)((char*)d_grads + uint64_t(idx[i]) * grad_value_size);
  }
}

template <typename GradType>
__global__ void merge_gradient_kernel(const uint32_t* offset,
                                      const uint32_t* fea_num,
                                      const uint32_t* index, GradType* input,
                                      GradType* output, int n,
                                      size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) {
    uint32_t start = offset[i];
    uint32_t num = fea_num[i];
    int ori_index = index[start];

    char* tmp_in = (char*)(input);
    char* tmp_out = (char*)(output);

    GradType& out_value = *((GradType*)(tmp_out + size_t(i) * grad_value_size));
    GradType& in_value  = *((GradType*)(tmp_in + size_t(ori_index) * grad_value_size));
    out_value = in_value;
    for (int j = 1; j < num; ++j) {
      ori_index = index[start + j];
      GradType& in_value_tmp  = *((GradType*)(tmp_in + size_t(ori_index) * grad_value_size));
      out_value += in_value_tmp;
    }
  }
}

__global__ void merge_key_info_kernel(const uint32_t* offset,
                                      const uint32_t* fea_num,
                                      const uint32_t* index, revert_info* input,
                                      revert_info* output, int n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) {
    uint32_t start = offset[i];
    uint32_t num = fea_num[i];
    int ori_index = index[start];

    output[i] = input[ori_index];

    for (int j = 1; j < num; ++j) {
      ori_index = index[start + j];
      if (output[i].key != input[ori_index].key || output[i].value_ptr != input[ori_index].value_ptr) {
        printf("merge key error \n");
      }
    }
  }
}

template <typename ValType, typename T>
__global__ void fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                           size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_vals[idx[i]] = d_shard_vals[i];
  }
}

template <typename ValType, typename T>
__global__ void dy_mf_fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                                 size_t len, size_t val_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    uint64_t new_offset = uint64_t(idx[i]) * val_size;
    *(ValType*)((char*)d_vals + new_offset) =
        *(ValType*)((char*)d_shard_vals + i * val_size);
  }
}

template <typename ValType, typename T>
__global__ void lxch_dy_mf_fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx, revert_info* d_revert_info,
                                 uint32_t* shard_idx, size_t len, size_t val_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    revert_info item_revert = d_revert_info[i];
    for (int j = 0; j < item_revert.len; j++) {
      ValType* des_value = (ValType*)(((char*)d_vals) + (idx[j + item_revert.offset]) * val_size);
      *des_value = *(ValType*)((char*)d_shard_vals + shard_idx[i] * val_size);
    }
  }
}

/*
template <typename ValType, typename T>
__global__ void lxch_dy_mf_fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx, revert_info* d_revert_info,
                                 uint32_t* shard_idx, size_t len, size_t val_size,
                                 uint64_t kernel_1_size, uint64_t kernel_8_size, uint64_t kernel_all_size) {
  const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = k / val_size;
  size_t m = k % val_size;
  if (i < kernel_all_size) {
    uint32_t cur_index = 0;
    uint32_t cur_offset = 0;
    uint32_t cur_batch = 0;
    if (i < kernel_1_size) {
      cur_index = i;
      cur_offset = 0;
      cur_batch = 1;
    } else if ( i < kernel_8_size) {
      cur_index = (i - kernel_1_size) / 8 + kernel_1_size;
      cur_offset = (i - kernel_1_size) % 8;
      cur_batch = 8;
    } else {
      cur_index = (i - kernel_8_size) / 32 + kernel_1_size + (kernel_8_size - kernel_1_size) / 8;
      cur_offset = (i - kernel_8_size) % 32;
      cur_batch = 32;
    }

    revert_info item_revert = d_revert_info[cur_index];
    for (int j = cur_offset; j < item_revert.len; j+=cur_batch) {
      char* des_value = (char*)d_vals + (idx[j + item_revert.offset]) * val_size +m;
      char* src_value = (char*)d_shard_vals + shard_idx[cur_index] * val_size;
      *des_value = *src_value;
    }
  }
}
*/

/*
template <typename ValType, typename T>
__global__ void lxch_dy_mf_fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx, revert_info* d_revert_info,
                                 uint32_t* shard_idx, size_t len, size_t val_size,
                                 uint64_t kernel_1_size, uint64_t kernel_8_size, uint64_t kernel_all_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kernel_all_size) {
    uint32_t cur_index = 0;
    uint32_t cur_offset = 0;
    uint32_t cur_batch = 0;
    if (i < kernel_1_size) {
      cur_index = i;
      cur_offset = 0;
      cur_batch = 1;
    } else if ( i < kernel_8_size) {
      cur_index = (i - kernel_1_size) / 8 + kernel_1_size;
      cur_offset = (i - kernel_1_size) % 8;
      cur_batch = 8;
    } else {
      cur_index = (i - kernel_8_size) / 32 + kernel_1_size + (kernel_8_size - kernel_1_size) / 8;
      cur_offset = (i - kernel_8_size) % 32;
      cur_batch = 32;
    }

    revert_info item_revert = d_revert_info[cur_index];
    for (int j = cur_offset; j < item_revert.len; j+=cur_batch) {
      ValType* des_value = (ValType*)(((char*)d_vals) + (idx[j + item_revert.offset]) * val_size);
      *des_value = *(ValType*)((char*)d_shard_vals + shard_idx[cur_index] * val_size);
    }
  }
}
*/

template <typename ValType, typename T>
__global__ void lxch_dy_mf_fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx, revert_info* d_revert_info,
                                 uint32_t* shard_idx, size_t len, size_t val_size,
                                 uint64_t kernel_1, uint64_t kernel_4, uint64_t kernel_16, uint64_t kernel_64, uint64_t kernel_256, size_t len_1) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t kernel_1_size = kernel_1;
  uint64_t kernel_4_size = kernel_1_size + kernel_4 * 4;
  uint64_t kernel_16_size = kernel_4_size + kernel_16 * 16;
  uint64_t kernel_64_size = kernel_16_size + kernel_64 * 64;
  uint64_t kernel_all_size = kernel_64_size + kernel_256 * 256;
  if (i < kernel_all_size) {
    uint32_t cur_index = 0;
    uint32_t cur_offset = 0;
    uint32_t cur_batch = 0;
    if (i < kernel_1_size) {
      cur_index = i;
      cur_offset = 0;
      cur_batch = 1;
    } else if (i < kernel_4_size) {
      cur_index = (i - kernel_1_size) / 4 + kernel_1;
      cur_offset = (i - kernel_1_size) % 4;
      cur_batch = 4;
    } else if (i < kernel_16_size) {
      cur_index = (i - kernel_4_size) / 16 + kernel_1 + kernel_4;
      cur_offset = (i - kernel_4_size) % 16;
      cur_batch = 16;
    } else if (i < kernel_64_size) {
      cur_index = (i - kernel_16_size) / 64 + kernel_1 + kernel_4 + kernel_16;
      cur_offset = (i - kernel_16_size) % 64;
      cur_batch = 64;
    } else {
      cur_index = (i - kernel_64_size) / 256 + kernel_1 + kernel_4 + kernel_16 + kernel_64;
      cur_offset = (i - kernel_64_size) % 256;
      cur_batch = 256;
    }
    if (cur_index >= len) {
      printf("lxchtestbb  %lu  %llu  %llu\n", cur_index, i, len);
      return;
    }
    revert_info item_revert = d_revert_info[cur_index];
    for (int j = cur_offset; j < item_revert.len; j+=cur_batch) {
      if (j + item_revert.offset >= len_1) {
        printf("lxchtestcc  %d  %llu  %llu\n", j + item_revert.offset, i, len_1);
        return;
      }
      if (idx[j + item_revert.offset] >= len_1) {
        printf("lxchtestdd  %lu  %llu  %llu\n", idx[j + item_revert.offset], i, len_1);
        return;
      }
      if (shard_idx[cur_index] >= len) {
        printf("lxchtestee  %lu  %llu  %llu\n", shard_idx[cur_index], i, len);
        return;
      }
      ValType* des_value = (ValType*)(((char*)d_vals) + (idx[j + item_revert.offset]) * val_size);
      *des_value = *(ValType*)((char*)d_shard_vals + shard_idx[cur_index] * val_size);
    }
  }
}

template <typename GradType>
__global__ void lxch_merge_gradient_kernel(GradType* output, GradType* input, uint32_t* sort_idx,
                                          revert_info* d_revert_info, uint32_t* shard_revert_idx, size_t len,
                                          size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < len) {
    revert_info item_revert = d_revert_info[i];
    char* tmp_in = (char*)(input);
    char* tmp_out = (char*)(output);
    GradType& out_value = *((GradType*)(tmp_out + shard_revert_idx[i] * grad_value_size));
    GradType& in_value  = *((GradType*)(tmp_in + sort_idx[item_revert.offset] * grad_value_size));
    out_value = in_value;
    for (int j = 1; j < item_revert.len; ++j) {
      GradType& in_value_tmp  = *((GradType*)(tmp_in + sort_idx[item_revert.offset + j] * grad_value_size));
      out_value += in_value_tmp;
    }
  }
}

template <typename GradType>
__global__ void lxch_merge_gradient_kernel(GradType* output, GradType* input, uint32_t* sort_idx,
                                          revert_info* d_revert_info, uint32_t* shard_revert_idx, size_t len,
                                          size_t grad_value_size,
                                          const uint64_t size_1, const uint64_t size_4, const uint64_t size_16, const uint64_t size_64, const uint64_t size_256,
                                          const uint64_t kernel_1, const uint64_t kernel_4, const uint64_t kernel_16, const uint64_t kernel_64, const uint64_t kernel_256) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kernel_256) {
    uint32_t cur_index = 0;
    uint32_t cur_offset = 0;
    uint32_t cur_batch = 0;
    if (i < kernel_1) {
      cur_index = i;
      cur_offset = 0;
      cur_batch = 1;
      if (cur_index >= size_1) {
        return;
      }
    } else if (i < kernel_4) {
      cur_index = (i - kernel_1) / 4;
      cur_offset = (i - kernel_1) % 4;
      cur_batch = 4;
      if (cur_index >= size_4) {
        return;
      }
      cur_index += size_1;
    } else if (i < kernel_16) {
      cur_index = (i - kernel_4) / 16;
      cur_offset = (i - kernel_4) % 16;
      cur_batch = 16;
      if (cur_index >= size_16) {
        return;
      }
      cur_index += size_1 + size_4;
    } else if (i < kernel_64) {
      cur_index = (i - kernel_16) / 64;
      cur_offset = (i - kernel_16) % 64;
      cur_batch = 64;
      if (cur_index >= size_64) {
        return;
      }
      cur_index += size_1 + size_4 + size_16;
    } else {
      cur_index = (i - kernel_64) / 256;
      cur_offset = (i - kernel_64) % 256;
      cur_batch = 256;
      if (cur_index >= size_256) {
        return;
      }
      cur_index += size_1 + size_4 + size_16 + size_64;
    }

    if (cur_batch == 1) {
      char* tmp_in = (char*)(input);
      char* tmp_out = (char*)(output);
      revert_info item_revert = d_revert_info[cur_index];
      GradType& out_value = *((GradType*)(tmp_out + shard_revert_idx[cur_index] * grad_value_size));
      GradType& in_value  = *((GradType*)(tmp_in + sort_idx[item_revert.offset] * grad_value_size));
      out_value = in_value;
      for (int j = 1; j < item_revert.len; ++j) {
        GradType& in_value_tmp  = *((GradType*)(tmp_in + sort_idx[item_revert.offset + j] * grad_value_size));
        out_value += in_value_tmp;
      }
    } else if (cur_batch <= 32) {
      auto g = cooperative_groups::this_thread_block();
      auto tile32 = cooperative_groups::tiled_partition(g, cur_batch);
      revert_info item_revert = d_revert_info[cur_index];
      char* tmp_in = (char*)(input);
      char* tmp_out = (char*)(output);
      uint32_t cur_len = item_revert.len;
      uint32_t now_batch = cur_batch;
      if (cur_len < cur_batch) {
        now_batch = 2;
        while (now_batch < cur_len) {
          now_batch *= 2;
        }
        now_batch /=2;
      }
      while(cur_len > 1) {
        for (int _j = now_batch; _j < cur_len; _j += now_batch) {
          int des_idx = cur_offset;
          int src_idx = des_idx + _j;
          if (src_idx < cur_len) {
            char* tmp_in = (char*)(input);
            GradType& out_value_tmp  = *((GradType*)(tmp_in + sort_idx[item_revert.offset + des_idx] * grad_value_size));
            GradType& in_value_tmp   = *((GradType*)(tmp_in + sort_idx[item_revert.offset + src_idx] * grad_value_size));
            out_value_tmp += in_value_tmp;
          }
        }
        if (cur_len > cur_batch) {
          cur_len = cur_batch;
          now_batch /= 2;
        } else {
          cur_len = now_batch;
          now_batch /= 2;
        }
        tile32.sync();
      }
      GradType& out_value = *((GradType*)(tmp_out + shard_revert_idx[cur_index] * grad_value_size));
      GradType& in_value  = *((GradType*)(tmp_in + sort_idx[item_revert.offset] * grad_value_size));
      out_value = in_value;
    } else {
      revert_info item_revert = d_revert_info[cur_index];
      char* tmp_in = (char*)(input);
      char* tmp_out = (char*)(output);
      uint32_t cur_len = item_revert.len;
      uint32_t now_batch = cur_batch;
      if (cur_len < cur_batch) {
        now_batch = 2;
        while (now_batch < cur_len) {
          now_batch *= 2;
        }
        now_batch /=2;
      }
      while(cur_len > 1) {
        for (int _j = now_batch; _j < cur_len; _j += now_batch) {
          int des_idx = cur_offset;
          int src_idx = des_idx + _j;
          if (src_idx < cur_len) {
            char* tmp_in = (char*)(input);
            GradType& out_value_tmp  = *((GradType*)(tmp_in + sort_idx[item_revert.offset + des_idx] * grad_value_size));
            GradType& in_value_tmp   = *((GradType*)(tmp_in + sort_idx[item_revert.offset + src_idx] * grad_value_size));
            out_value_tmp += in_value_tmp;
          }
        }
        if (cur_len > cur_batch) {
          cur_len = cur_batch;
          now_batch /= 2;
        } else {
          cur_len = now_batch;
          now_batch /= 2;
        }
        __syncthreads();
      }
      GradType& out_value = *((GradType*)(tmp_out + shard_revert_idx[cur_index] * grad_value_size));
      GradType& in_value  = *((GradType*)(tmp_in + sort_idx[item_revert.offset] * grad_value_size));
      out_value = in_value;
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::lxch_barrir() {
  uint32_t total_gpu = resource_->total_gpu();
  uint64_t cur_barrir_2_ = barrir_2_.load();
  cur_barrir_2_ = (cur_barrir_2_ + total_gpu - 1) / total_gpu * total_gpu;
  barrir_1_.fetch_add(1);
  while (barrir_1_.load() != cur_barrir_2_) {
    continue;
  }
  barrir_2_.fetch_add(1);
  while (barrir_2_.load() % resource_->total_gpu() != 0) {
    continue;
  }
  return;
}

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::HeterComm(
    size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  
//  LxchDebug::get_ins()->lxch_test_set.clear();
  VLOG(1) << "Construct new HeterComm";
  resource_ = resource;
  barrir_1_.store(0);
  barrir_2_.store(resource_->total_gpu());
  storage_.resize(resource_->total_gpu());
  split_key_info_.resize(resource_->total_gpu());
  multi_mf_dim_ = resource->multi_mf();
  for (int i = 0; i < resource_->total_gpu(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    // allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
    //     2, 1, 20, (size_t)-1, false, false));  // NOLINT
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        8, 1, (unsigned int)-1, (size_t)-1, false, false));
    if (!multi_mf_dim_) {
      auto table = new Table(capacity / load_factor_);
      tables_.push_back(table);
    } else {
      max_mf_dim_ = resource->max_mf_dim();
      size_t val_type_size = g_transfor->get_gpu_value_size(max_mf_dim_);
      size_t grad_type_size = g_transfor->get_gpu_push_value_size(max_mf_dim_);
      auto ptr_table = new PtrTable(capacity / load_factor_);
      ptr_table->set_feature_value_size(val_type_size, grad_type_size);
      ptr_tables_.push_back(ptr_table);
    }
    if (multi_node_) {
      storage_[i].init(feanum_, resource_->dev_id(i));
    }
  }
  init_path();
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::init_path() {
  int total_gpu = resource_->total_gpu();
  path_.resize(total_gpu);

  if (!topo_aware_) {
    VLOG(0) << "init path without topo aware";
    for (int i = 0; i < total_gpu; ++i) {
      path_[i].resize(total_gpu);
      for (int j = 0; j < total_gpu; ++j) {
        auto& nodes = path_[i][j].nodes_;
        nodes.resize(1);
        nodes[0].in_stream = resource_->comm_stream(i, j);
        nodes[0].out_stream = resource_->comm_stream(i, j);
        nodes[0].key_storage = NULL;
        nodes[0].val_storage = NULL;
        nodes[0].sync = 0;
        nodes[0].gpu_num = j;
      }
    }
  } else {
    VLOG(0) << "init path with topo aware";
    for (int i = 0; i < total_gpu; ++i) {
      path_[i].resize(total_gpu);
      for (int j = 0; j < total_gpu; ++j) {
        auto& nodes = path_[i][j].nodes_;
        int from = resource_->dev_id(i);
        int to = resource_->dev_id(j);
        int transfer_id = i;
        if (need_transfer(from, to)) {
          transfer_id = resource_->get_index_by_devid(get_transfer_devid(from));
          nodes.push_back(Node());
          Node& node = nodes.back();
          node.in_stream = resource_->comm_stream(i, transfer_id);
          node.out_stream = resource_->comm_stream(transfer_id, i);
          node.key_storage = NULL;
          node.val_storage = NULL;
          node.sync = 1;
          node.gpu_num = transfer_id;
        }
        nodes.push_back(Node());
        Node& node = nodes.back();
        node.in_stream = resource_->comm_stream(i, transfer_id);
        node.out_stream = resource_->comm_stream(transfer_id, i);
        node.key_storage = NULL;
        node.val_storage = NULL;
        node.sync = 0;
        node.gpu_num = j;
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::create_storage(int start_index,
                                                           int end_index,
                                                           size_t keylen,
                                                           size_t vallen) {
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].gpu_num));
    platform::CUDAPlace place = platform::CUDAPlace(resource_->dev_id(nodes[i].gpu_num));
/*    
    if (keylen != 0) {
      if (nodes[i].d_key_storage == NULL || nodes[i].d_key_storage->size() < keylen) {
        nodes[i].d_key_storage = NULL;
        nodes[i].d_key_storage = memory::Alloc(place, keylen);
      }
      nodes[i].key_bytes_len = keylen;
      nodes[i].key_storage = (char*)nodes[i].d_key_storage->ptr();
    }
    if (vallen != 0) {
      if (nodes[i].d_val_storage == NULL || nodes[i].d_val_storage->size() < vallen) {
        nodes[i].d_val_storage = NULL;
        nodes[i].d_val_storage = memory::Alloc(place, vallen);
      }
      nodes[i].val_bytes_len = vallen;
      nodes[i].val_storage = (char*)nodes[i].d_val_storage->ptr();
    }
*/  
    if (keylen != 0) {
      if (nodes[i].key_storage != nullptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(resource_->dev_id(nodes[i].gpu_num),
                            nodes[i].key_storage));
        nodes[i].key_storage = nullptr;
      }
      PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
          resource_->dev_id(nodes[i].gpu_num),
          (void**)&(nodes[i].key_storage),  // NOLINT
          keylen, resource_->remote_stream(nodes[i].gpu_num, start_index)));
      nodes[i].key_bytes_len = keylen;
    }
    if (vallen != 0) {
      if (nodes[i].val_storage != nullptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(resource_->dev_id(nodes[i].gpu_num),
                            nodes[i].val_storage));
        nodes[i].val_storage = nullptr;
      }
      PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
          resource_->dev_id(nodes[i].gpu_num),
          (void**)&(nodes[i].val_storage),  // NOLINT
          vallen, resource_->remote_stream(nodes[i].gpu_num, start_index)));
      nodes[i].val_bytes_len = vallen;
    }
    
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::destroy_storage(int start_index,
                                                            int end_index) {
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].gpu_num));
/*    
    nodes[i].key_storage = nullptr;
    nodes[i].val_storage = nullptr;
    nodes[i].d_key_storage = NULL;
    nodes[i].d_val_storage = NULL;
*/    
    
    if (nodes[i].key_storage != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(resource_->dev_id(nodes[i].gpu_num),
                            nodes[i].key_storage));
      nodes[i].key_storage = nullptr;
    }
    if (nodes[i].val_storage != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(resource_->dev_id(nodes[i].gpu_num),
                            nodes[i].val_storage));
      nodes[i].val_storage = nullptr;
    }
    
  }
}

template <typename KeyType, typename ValType, typename GradType>
template <typename KeyInfo>
void HeterComm<KeyType, ValType, GradType>::walk_to_dest(
    int start_index, int gpu_num, int* h_left, int* h_right, KeyInfo* src_key,
    GradType* src_val) {
  int need_copy_val = 0;
  if (src_val) {
    need_copy_val = 1;
  }
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int size = path_[start_index][i].nodes_.size();
    auto& node = path_[start_index][i].nodes_[0];
    CopyTask t(&path_[start_index][i], 0);
    que.push(t);
    cudaMemcpyAsync(node.key_storage,
                    reinterpret_cast<char*>(src_key + h_left[i]),
                    node.key_bytes_len, cudaMemcpyDefault, node.in_stream);
    if (need_copy_val) {
      cudaMemcpyAsync(node.val_storage,
                      reinterpret_cast<char*>(src_val + h_left[i]),
                      node.val_bytes_len, cudaMemcpyDefault, node.in_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_task.step].in_stream);
    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].key_storage,
                      cur_task.path->nodes_[cur_step].key_storage,
                      cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step + 1].in_stream);
      if (need_copy_val) {
        cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].val_storage,
                        cur_task.path->nodes_[cur_step].val_storage,
                        cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                        cudaMemcpyDefault,
                        cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
template <typename KeyInfo>
void HeterComm<KeyType, ValType, GradType>::walk_to_dest(
    int start_index, int gpu_num, int* h_left, int* h_right, KeyInfo* src_key,
    char* src_val, size_t val_size) {
  int need_copy_val = 0;
  if (src_val) {
    need_copy_val = 1;
  }
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int size = path_[start_index][i].nodes_.size();
    auto& node = path_[start_index][i].nodes_[0];
    CopyTask t(&path_[start_index][i], 0);
    que.push(t);
    if (src_key != nullptr) {
      cudaMemcpyAsync(node.key_storage,
                      reinterpret_cast<char*>(src_key + h_left[i]),
                      node.key_bytes_len, cudaMemcpyDefault, node.in_stream);
    }
    if (need_copy_val) {
      cudaMemcpyAsync(node.val_storage,
                      src_val + uint64_t(h_left[i]) * uint64_t(val_size),
                      node.val_bytes_len, cudaMemcpyDefault, node.in_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_task.step].in_stream);
    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);
      if (src_key != nullptr) {
        cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].key_storage,
                        cur_task.path->nodes_[cur_step].key_storage,
                        cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                        cudaMemcpyDefault,
                        cur_task.path->nodes_[cur_step + 1].in_stream);
      }
      if (need_copy_val) {
        cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].val_storage,
                        cur_task.path->nodes_[cur_step].val_storage,
                        cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                        cudaMemcpyDefault,
                        cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_src(
    int start_index, int gpu_num, int* h_left, int* h_right, ValType* src_val) {
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[i]),
                      node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
                      node.out_stream);
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
      cudaMemcpyAsync(path_[start_index][i].nodes_[cur_step - 1].val_storage,
                      node.val_storage,
                      path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      path_[start_index][i].nodes_[cur_step - 1].out_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_step].out_stream);
    }
    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step - 1].out_stream);
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().gpu_num;
      cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[end_index]),
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step].out_stream);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::walk_to_src(
    int start_index, int gpu_num, int* h_left, int* h_right, char* src_val, size_t val_size) {
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      cudaMemcpyAsync(src_val + uint64_t(h_left[i]) * val_size,
                      node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
                      node.out_stream);
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
      cudaMemcpyAsync(path_[start_index][i].nodes_[cur_step - 1].val_storage,
                      node.val_storage,
                      path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      path_[start_index][i].nodes_[cur_step - 1].out_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_step].out_stream);
    }
    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step - 1].out_stream);
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().gpu_num;
      cudaMemcpyAsync(src_val + uint64_t(h_left[end_index]) * val_size,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step].out_stream);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
HeterComm<KeyType, ValType, GradType>::~HeterComm() {
  if (!multi_mf_dim_) {
    for (auto& table : tables_) {
      delete table;
      table = nullptr;
    }
  } else {
    for (auto& table : ptr_tables_) {
      delete table;
      table = nullptr;
    }
    for (auto& table : tables_) {
      delete table;
      table = nullptr;
    }
   }
  
  int total_gpu = resource_->total_gpu();
  for (size_t i = 0; i < total_gpu; i++) {
    auto& allocator = allocators_[i];
    int dev_id = resource_->dev_id(i);
    platform::CUDADeviceGuard guard(dev_id);
    for (int j = 0; j < total_gpu; ++j) {
      destroy_storage(i, j);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::show_one_table(int gpu_num) {
  if (!multi_mf_dim_) {
    tables_[gpu_num]->show();
  } else {
    // ptr_tables_[gpu_num]->show();
  }
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::log2i(int x) {
  unsigned res = 0;
  while (x >>= 1) {
    ++res;
  }
  return res;
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::get_index_by_devid(int devid) {
  return resource_->get_index_by_devid(devid);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::build_ps(int num, KeyType* h_keys,
                                                     ValType* h_vals,
                                                     size_t len,
                                                     size_t chunk_size,
                                                     int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  std::vector<memory::allocation::AllocationPtr> d_key_bufs;
  std::vector<memory::allocation::AllocationPtr> d_val_bufs;

  gpuStream_t streams[stream_num];  // NOLINT
  for (int i = 0; i < stream_num; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&(streams[i])));
    auto d_k_buf = memory::Alloc(place, chunk_size * sizeof(KeyType));
    auto d_v_buf = memory::Alloc(place, chunk_size * sizeof(ValType));
    d_key_bufs.push_back(std::move(d_k_buf));
    d_val_bufs.push_back(std::move(d_v_buf));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_key_bufs[cur_stream]->ptr(), h_keys + cur_len,
                        sizeof(KeyType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_val_bufs[cur_stream]->ptr(), h_vals + cur_len,
                        sizeof(ValType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    tables_[num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<ValType*>(d_val_bufs[cur_stream]->ptr()), tmp_len,
        streams[cur_stream]);
    cur_stream += 1;
    cur_len += tmp_len;
  }

  for (int i = 0; i < stream_num; ++i) {
    cudaStreamSynchronize(streams[i]);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(streams[i]));
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::build_ps(int num, KeyType* h_keys,
                                                     char* pool,
                                                     size_t len,
                                                     size_t feature_value_size,
                                                     size_t chunk_size,
                                                     int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  /*
  for (size_t kk = 0; kk < len; kk++) {
    VLOG(0) << " " << kk << " " << num << " cpu key is: " << h_keys[kk] << "  idx:" << num;
  }
  
  mutex_.lock();
  for (size_t kk = 0; kk < len; kk++) {
    LxchDebug::get_ins()->lxch_test_set.insert(h_keys[kk]);
  }
  mutex_.unlock();
  */

  // use hbm pool
  std::vector<memory::allocation::AllocationPtr> d_key_bufs;
  std::vector<memory::allocation::AllocationPtr> h_pinned_key_bufs;
  gpuStream_t streams[stream_num];
  for (int i = 0; i < stream_num; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&(streams[i])));
    auto d_k_buf = memory::Alloc(place, chunk_size * sizeof(KeyType));
    d_key_bufs.push_back(std::move(d_k_buf));
    auto h_k_buf = memory::Alloc(phi::GPUPinnedPlace(), chunk_size * sizeof(KeyType));
    h_pinned_key_bufs.push_back(std::move(h_k_buf));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    cudaStreamSynchronize(streams[cur_stream]);
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
    memcpy(h_pinned_key_bufs[cur_stream]->ptr(),  h_keys + cur_len, sizeof(KeyType) * tmp_len);
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_key_bufs[cur_stream]->ptr(), h_pinned_key_bufs[cur_stream]->ptr(),
                        sizeof(KeyType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    ptr_tables_[num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()), tmp_len,
        pool, feature_value_size, cur_len, streams[cur_stream]);
    cur_stream += 1;
    cur_len += tmp_len;
  }

  for (int i = 0; i < stream_num; ++i) {
    cudaStreamSynchronize(streams[i]);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(streams[i]));
  }
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::merge_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    int& uniq_len) {  // NOLINT
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());

  auto d_merge_grads = memory::Alloc(place, len * sizeof(GradType));
  GradType* d_merge_grads_ptr =
      reinterpret_cast<GradType*>(d_merge_grads->ptr());

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_grads,
      d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));

  void* d_buff = NULL;
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_grads, d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));
  temp_storage_bytes = 0;

  auto d_num_runs_out_mem = memory::Alloc(place, sizeof(int));
  int* d_num_runs_out = reinterpret_cast<int*>(d_num_runs_out_mem->ptr());

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
      NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_merge_grads_ptr,
      d_grads, d_num_runs_out, merger_, len, stream, false));

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_merge_grads_ptr, d_grads, d_num_runs_out, merger_, len, stream, false));

  cudaMemcpyAsync(&uniq_len, d_num_runs_out, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::merge_grad(int gpu_num,
                                                       KeyType* d_keys,
                                                       GradType* d_grads,
                                                       float* mf, size_t len,
                                                       int& uniq_len) {
  platform::Timer timeline;
  timeline.Start();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;

  //VLOG(1) << "hetercomm merge_grad: max_mf_dim: " << max_mf_dim_;
  size_t grad_value_size = g_transfor->get_gpu_push_value_size(max_mf_dim_);

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());

  auto d_merge_grads = memory::Alloc(place, len * grad_value_size);
  GradType* d_merge_grads_ptr =
      reinterpret_cast<GradType*>(d_merge_grads->ptr());

  auto d_fea_num_info =
      memory::Alloc(place, sizeof(uint32_t) * (len * 3 + 1));
  uint32_t* d_fea_num_info_ptr =
      reinterpret_cast<uint32_t*>(d_fea_num_info->ptr());
  uint32_t* d_index = (uint32_t*)&d_fea_num_info_ptr[len];
  uint32_t* d_idx = (uint32_t*)&d_index[len];
  int* d_merged_size = (int*)&d_idx[len];
  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx, len);
  //step1 对key做排序, 
  //d_keys: 原始key列表
  //d_merge_keys_ptr: 排序后的列表
  //d_idx: 原始编号
  //d_index: 排序后的编号， 也就是d_merge_keys_ptr元素中每个key在原始d_keys中的哪一个位置
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_idx, d_index, len,
      0, 8 * sizeof(KeyType), stream));

  void* d_buff = NULL;
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_idx, d_index, len, 0, 8 * sizeof(KeyType), stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  timeline.Pause();
  timeline.Start();
  //step2，每个key总共重复了多少次
  //d_merge_keys_ptr: 原始输入key列表
  //d_keys: 去重后的key列表
  //d_fea_num_info_ptr: 对应的key的重复元素个数
  //d_merged_size: 总计有多少个去重key
  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_fea_num_info_ptr,
      d_merged_size, len, stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_fea_num_info_ptr, d_merged_size, len, stream));

  cudaMemcpyAsync((void*)&uniq_len, d_merged_size, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  timeline.Pause();
  timeline.Start();

  assert(d_merged_size > 0);
  uint32_t* d_offset = (uint32_t*)&d_index[len];

  //step3 做前缀规约
  //d_fea_num_info_ptr: 输入的key的重复元数个数
  //d_offset: 前缀规约结果
  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, d_fea_num_info_ptr, d_offset, uniq_len,
      stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      d_temp_storage->ptr(), temp_storage_bytes, d_fea_num_info_ptr, d_offset,
      uniq_len, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  timeline.Pause();
  timeline.Start();
  grid_size = (uniq_len - 1) / block_size_ + 1;
  merge_gradient_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_offset, d_fea_num_info_ptr, d_index, (GradType*)d_grads,
      (GradType*)d_merge_grads_ptr, uniq_len, grad_value_size);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  timeline.Pause();
  timeline.Start();

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(d_grads, d_merge_grads_ptr, grad_value_size * uniq_len,
                      cudaMemcpyDeviceToDevice, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  timeline.Pause();
  timeline.Start();
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::split_input_to_shard(
    KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right,
    int gpu_num) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto d_idx_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_idx_tmp_ptr = reinterpret_cast<int*>(d_idx_tmp->ptr());

  auto d_shard_index = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_ptr = reinterpret_cast<int*>(d_shard_index->ptr());

  auto d_shard_index_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_tmp_ptr = reinterpret_cast<int*>(d_shard_index_tmp->ptr());

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);
  calc_shard_index<<<grid_size, block_size_, 0, stream>>>(
      d_keys, len, d_shard_index_tmp_ptr, total_gpu);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_gpu);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));
  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(d_shard_index_ptr,
                                                           left, right, len);
  cudaStreamSynchronize(stream);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::lxch_split_input_to_shard(
    revert_info* d_keys, int* d_idx_ptr, size_t len, int* left, int* right,
    int gpu_num) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto d_idx_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_idx_tmp_ptr = reinterpret_cast<int*>(d_idx_tmp->ptr());

  auto d_shard_index = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_ptr = reinterpret_cast<int*>(d_shard_index->ptr());

  auto d_shard_index_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_tmp_ptr = reinterpret_cast<int*>(d_shard_index_tmp->ptr());

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);
  calc_shard_index<<<grid_size, block_size_, 0, stream>>>(
      d_keys, len, d_shard_index_tmp_ptr, total_gpu);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_gpu);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));
  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(d_shard_index_ptr,
                                                           left, right, len);
  cudaStreamSynchronize(stream);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::lxch_pull_sparse(int num,
                                                        KeyType* d_keys,
                                                        ValType* d_vals,
                                                        size_t len) {
  if (len == 0) {
    return;
  }
  
//  lxch_barrir();
//  auto lxch_step_1 = platform::Timer::lxch_get_base_time();
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);
  int grid_size = (len - 1) / block_size_ + 1;

  for (int i = 0; i < total_gpu; ++i) {
    destroy_storage(num, i);
  }

  //step1，资源分配
  auto& split_info = split_key_info_[num];
  auto d_src_idxs = memory::Alloc(place, len * sizeof(uint32_t));
  uint32_t* d_src_idxs_ptr = (uint32_t*)(d_src_idxs->ptr());
  auto d_sort_keys = memory::Alloc(place, len * sizeof(uint64_t));
  uint64_t* d_sort_keys_ptr = (uint64_t*)(d_sort_keys->ptr());
  auto d_uniq_keys = memory::Alloc(place, len * sizeof(uint64_t));
  uint64_t* d_uniq_keys_ptr = (uint64_t*)(d_uniq_keys->ptr());
  uint32_t* d_uniq_size_ptr = d_src_idxs_ptr;
  auto d_uniq_key_size = memory::Alloc(place, len * sizeof(uint32_t));
  uint32_t* d_uniq_key_size_ptr = (uint32_t*)(d_uniq_key_size->ptr());
  if (split_info.d_sort_idxs == nullptr || split_info.d_sort_idxs->size() < len * sizeof(uint32_t)) {
    split_info.d_sort_idxs = NULL;
    split_info.d_sort_idxs = memory::Alloc(place, len * sizeof(uint32_t));
  }
  uint32_t* d_sort_idxs_ptr = (uint32_t*)(split_info.d_sort_idxs->ptr());
  if (split_info.h_uniq_size == nullptr || split_info.h_uniq_size->size() < sizeof(uint32_t)) {
    split_info.h_uniq_size = NULL;
    split_info.h_uniq_size = memory::Alloc(phi::GPUPinnedPlace(), sizeof(uint32_t));
  }

//  auto lxch_step_2 = platform::Timer::lxch_get_base_time();
  //step2 做去重
  //step2.1 填充索引
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_src_idxs_ptr, len);


//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_3 = platform::Timer::lxch_get_base_time();

  //step2.2 按key排序
  size_t temp_storage_bytes;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_sort_keys_ptr, d_src_idxs_ptr, 
      d_sort_idxs_ptr, len, 0, 8 * sizeof(KeyType), stream));
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_sort_keys_ptr, d_src_idxs_ptr, 
      d_sort_idxs_ptr, len, 0, 8 * sizeof(KeyType), stream));

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_4 = platform::Timer::lxch_get_base_time();


  //step2.3 然后去重
  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      NULL, temp_storage_bytes, d_sort_keys_ptr, d_uniq_keys_ptr, d_uniq_key_size_ptr,
      d_uniq_size_ptr, len, stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      d_temp_storage->ptr(), temp_storage_bytes, d_sort_keys_ptr, d_uniq_keys_ptr, 
      d_uniq_key_size_ptr, d_uniq_size_ptr, len, stream));
  

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_5 = platform::Timer::lxch_get_base_time();


  //step2.4 将去重后的数量导出出来
  uint32_t* h_uniq_size = reinterpret_cast<uint32_t*>(split_info.h_uniq_size->ptr());
  cudaMemcpyAsync((void*)h_uniq_size, d_uniq_size_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_6 = platform::Timer::lxch_get_base_time();


  //step3 去重后的信息分配资源
  uint32_t* d_len_offset_ptr = d_uniq_size_ptr;
  int* d_normal_size_ptr = (int*)d_uniq_size_ptr;
  auto d_revert_info = memory::Alloc(place, (*h_uniq_size) * sizeof(revert_info));
  revert_info* d_revert_info_ptr = (revert_info*)(d_revert_info->ptr());
  if (split_info.d_revert_info_1 == nullptr || split_info.d_revert_info_1->size() < (*h_uniq_size) * sizeof(revert_info)) {
    split_info.d_revert_info_1 = NULL;
    split_info.d_revert_info_1 = memory::Alloc(place, (*h_uniq_size) * sizeof(revert_info));
  }
  revert_info* d_revert_info_1_ptr = (revert_info*)(split_info.d_revert_info_1->ptr());
  if (split_info.d_uniq_size_1 == nullptr || split_info.d_uniq_size_1->size() < (*h_uniq_size) * sizeof(uint32_t)) {
    split_info.d_uniq_size_1 = NULL;
    split_info.d_uniq_size_1 = memory::Alloc(place, (*h_uniq_size) * sizeof(uint32_t));
  }
  uint32_t* d_uniq_size_1_ptr = (uint32_t*)(split_info.d_uniq_size_1->ptr());
  if (split_info.h_shard_left == nullptr || split_info.h_shard_left->size() < sizeof(uint32_t) * total_gpu) {
    split_info.h_shard_left = NULL;
    split_info.h_shard_left = memory::Alloc(phi::GPUPinnedPlace(), sizeof(uint32_t) * total_gpu);
  }
  if (split_info.h_shard_right == nullptr || split_info.h_shard_right->size() < sizeof(uint32_t) * total_gpu) {
    split_info.h_shard_right = NULL;
    split_info.h_shard_right = memory::Alloc(phi::GPUPinnedPlace(), sizeof(uint32_t) * total_gpu);
  }
  if (split_info.h_normal_size == nullptr || split_info.h_normal_size->size() < 5 * sizeof(int)) {
    split_info.h_normal_size = NULL;
    split_info.h_normal_size = memory::Alloc(phi::GPUPinnedPlace(), 5 * sizeof(int));
  }

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_7 = platform::Timer::lxch_get_base_time();


  //step4 建立反向映射关系
  //step4.1 进行前缀规约
  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, d_uniq_key_size_ptr, d_len_offset_ptr, *h_uniq_size,
      stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      d_temp_storage->ptr(), temp_storage_bytes, d_uniq_key_size_ptr, 
      d_len_offset_ptr, *h_uniq_size, stream));
  
//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_8 = platform::Timer::lxch_get_base_time();
  
  //step4.2 填充映射信息
  grid_size = ((*h_uniq_size) - 1) / block_size_ + 1;
  fill_revert_idx<<<grid_size, block_size_, 0, stream>>>(d_uniq_keys_ptr, d_uniq_key_size_ptr,
                                d_len_offset_ptr, d_revert_info_ptr, *h_uniq_size);
  
//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_9 = platform::Timer::lxch_get_base_time();

  //step4.3 反向按照key的数量进行排序
  temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_uniq_key_size_ptr, d_uniq_size_1_ptr, d_revert_info_ptr, 
      d_revert_info_1_ptr, *h_uniq_size, 0, 8 * sizeof(KeyType), stream));
  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_uniq_key_size_ptr, d_uniq_size_1_ptr, d_revert_info_ptr, 
      d_revert_info_1_ptr, *h_uniq_size, 0, 8 * sizeof(KeyType), stream));
  
//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_10 = platform::Timer::lxch_get_base_time();
  
  //step5 数据切割，将数据切分到不同的gpu卡里面取
  //先进行数据切割
  int* h_left = reinterpret_cast<int*>(split_info.h_shard_left->ptr());
  int* h_right = reinterpret_cast<int*>(split_info.h_shard_right->ptr());
  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());
  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  auto d_idx = memory::Alloc(place, (*h_uniq_size) * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  size_t val_type_size = g_transfor->get_gpu_value_size(max_mf_dim_);

  auto d_shard_keys = memory::Alloc(place, (*h_uniq_size) * sizeof(revert_info));
  revert_info* d_shard_keys_ptr = reinterpret_cast<revert_info*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::Alloc(place, (*h_uniq_size) * val_type_size);
  ValType* d_shard_vals_ptr = reinterpret_cast<ValType*>(d_shard_vals->ptr());

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_11 = platform::Timer::lxch_get_base_time();

  lxch_split_input_to_shard(d_revert_info_1_ptr, d_idx_ptr, (*h_uniq_size), d_left_ptr, d_right_ptr, num);

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_12 = platform::Timer::lxch_get_base_time();

  fill_shard_key<<<grid_size, block_size_, 0, stream>>>(d_shard_keys_ptr,
                                                        d_revert_info_1_ptr, d_idx_ptr, (*h_uniq_size));
  
//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_13 = platform::Timer::lxch_get_base_time();
  
  cudaMemcpyAsync(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemsetAsync(d_normal_size_ptr, 0, 5 * sizeof(uint32_t), stream);
  compute_normal_size<<<grid_size, block_size_, 0, stream>>>(d_uniq_size_1_ptr, d_normal_size_ptr, *h_uniq_size);
  cudaMemcpyAsync(split_info.h_normal_size->ptr(), d_normal_size_ptr, 5 * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_14 = platform::Timer::lxch_get_base_time();

  //分配对应的资源
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(num, i, shard_len * sizeof(revert_info),
                   shard_len * val_type_size);
  }

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_15 = platform::Timer::lxch_get_base_time();

  walk_to_dest(num, total_gpu, h_left, h_right, d_shard_keys_ptr, NULL);

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_16 = platform::Timer::lxch_get_base_time();

  std::vector<platform::Timer> time_lines;
  time_lines.resize(total_gpu);

  for (int i = 0; i < total_gpu; ++i) {
    time_lines[i].Start();
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    {


      ptr_tables_[i]->rwlock_->RDLock();
      ptr_tables_[i]->get(resource_->remote_stream(i, num),
                          reinterpret_cast<revert_info*>(node.key_storage),
                          reinterpret_cast<ValType*>(node.val_storage), h_right[i] - h_left[i] + 1, i);
    }
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    if (!multi_mf_dim_) {
      tables_[i]->rwlock_->UNLock();
    } else {
      ptr_tables_[i]->rwlock_->UNLock();
    }
    time_lines[i].Pause();
  }

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_17 = platform::Timer::lxch_get_base_time();

  walk_to_src(num, total_gpu, h_left, h_right, reinterpret_cast<char*>(d_shard_vals_ptr), val_type_size);


  for (int i = 0; i < total_gpu; ++i) {
    auto& node = path_[num][i].nodes_.front();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(node.out_stream));
  }

//  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_18 = platform::Timer::lxch_get_base_time();

  fill_reserve_shard_inx<<<grid_size, block_size_, 0, stream>>>(d_uniq_size_1_ptr, (uint32_t*)d_idx_ptr, *h_uniq_size);

//  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
//  auto lxch_step_19 = platform::Timer::lxch_get_base_time();

  int* normal_size = (int*)split_info.h_normal_size->ptr();
  normal_size[0] -= 1;
  normal_size[1] -= 1;
  normal_size[2] -= 1;
  normal_size[3] -= 1;
  normal_size[4] -= 1;
  if (normal_size[4] == -1) normal_size[4] = (*h_uniq_size);
  if (normal_size[3] == -1) normal_size[3] = normal_size[4];
  if (normal_size[2] == -1) normal_size[2] = normal_size[3];
  if (normal_size[1] == -1) normal_size[1] = normal_size[2];
  if (normal_size[0] == -1) normal_size[0] = normal_size[1];
  uint64_t kernel_1  = normal_size[1] - normal_size[0] > 0 ?  normal_size[1] - normal_size[0] : 0;
  uint64_t kernel_4  = normal_size[2] - normal_size[1] > 0 ?  normal_size[2] - normal_size[1] : 0;
  uint64_t kernel_16  = normal_size[3] - normal_size[2] > 0 ?  normal_size[3] - normal_size[2] : 0;
  uint64_t kernel_64  = normal_size[4] - normal_size[3] > 0 ?  normal_size[4] - normal_size[3] : 0;
  uint64_t kernel_256 = (*h_uniq_size) - normal_size[4] > 0 ?  (*h_uniq_size) - normal_size[4] : 0;
  /*
  uint64_t kernel_all = kernel_1 + kernel_8 * 8 + kernel_32 * 32;
  grid_size = (kernel_all - 1) / block_size_ + 1;
  lxch_dy_mf_fill_dvals<<<grid_size, block_size_, 0, stream>>>(d_shard_vals_ptr, d_vals, d_sort_idxs_ptr,
                            d_revert_info_1_ptr, d_uniq_size_1_ptr, *h_uniq_size, val_type_size,
                            kernel_1, kernel_1 + kernel_8 * 8, kernel_all);
  */
  uint64_t kernel_all = kernel_1 + kernel_4 * 4 + kernel_16 * 16 + kernel_64 * 64 + kernel_256 * 256;
  grid_size = (kernel_all - 1) / block_size_ + 1;
  lxch_dy_mf_fill_dvals<<<grid_size, block_size_, 0, stream>>>(d_shard_vals_ptr, d_vals, d_sort_idxs_ptr,
                            d_revert_info_1_ptr, d_uniq_size_1_ptr, *h_uniq_size, val_type_size,
                            kernel_1, kernel_4, kernel_16, kernel_64, kernel_256, len);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));


/*  LXCH_PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  auto lxch_step_20 = platform::Timer::lxch_get_base_time();
  VLOG(0) << "lxch pull detail: " << " device:" << num
                                  << " sa:" << lxch_step_2 - lxch_step_1
                                  << " sb:" << lxch_step_3 - lxch_step_2
                                  << " sc:" << lxch_step_4 - lxch_step_3
                                  << " sd:" << lxch_step_5 - lxch_step_4
                                  << " se:" << lxch_step_6 - lxch_step_5
                                  << " sf:" << lxch_step_7 - lxch_step_6
                                  << " sg:" << lxch_step_8 - lxch_step_7
                                  << " sh:" << lxch_step_9 - lxch_step_8
                                  << " si:" << lxch_step_10 - lxch_step_9
                                  << " sj:" << lxch_step_11 - lxch_step_10
                                  << " sk:" << lxch_step_12 - lxch_step_11
                                  << " sl:" << lxch_step_13 - lxch_step_12
                                  << " sm:" << lxch_step_14 - lxch_step_13
                                  << " sn:" << lxch_step_15 - lxch_step_14
                                  << " so:" << lxch_step_16 - lxch_step_15
                                  << " sp:" << lxch_step_17 - lxch_step_16
                                  << " sq:" << lxch_step_18 - lxch_step_17
                                  << " sr:" << lxch_step_19 - lxch_step_18
                                  << " ss:" << lxch_step_20 - lxch_step_19;
*/

  /*
  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(num, i);
  }
  */


  if (0) {
    //test
    KeyType* h_src_key = new KeyType[len];
    cudaMemcpy(h_src_key, d_keys, len * sizeof(KeyType), cudaMemcpyDeviceToHost);

    uint32_t* h_sort_idx = new uint32_t[len];
    cudaMemcpy(h_sort_idx, d_sort_idxs_ptr, len * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    revert_info* h_key_info = new revert_info[*h_uniq_size];
    cudaMemcpy(h_key_info, d_revert_info_1_ptr, (*h_uniq_size) * sizeof(revert_info), cudaMemcpyDeviceToHost);

    KeyType* h_src_key_1 = new KeyType[len];
    for (int i = 0; i < *h_uniq_size; i++) {
      for (int j = h_key_info[i].offset; j < h_key_info[i].offset + h_key_info[i].len; j++) {
        h_src_key_1[h_sort_idx[j]] = h_key_info[i].key;
      }
    }

    for (int i = 0; i < len; i++) {
      if (h_src_key[i] != h_src_key_1[i]) {
        abort();
      }
    }
  }

}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::pull_sparse(int num,
                                                        KeyType* d_keys,
                                                        ValType* d_vals,
                                                        size_t len) {

  /*
  size_t total_size = 0;
  size_t free_size = 0;
  cudaMemGetInfo(&free_size, &total_size);
  VLOG(0) << "lxch cudamemory1 device_id:" << num << "  " << "  memory:" << free_size / 1024/1024/1204;
  */
  auto lxch_step_1 = platform::Timer::lxch_get_base_time();

  /*
  auto lxch_tmp = memory::Alloc(phi::GPUPinnedPlace(), len * sizeof(KeyType));
  cudaMemcpy((void*)lxch_tmp->ptr(), d_keys, len * sizeof(KeyType), cudaMemcpyDeviceToHost);
  KeyType* lxch_tmp_1 = (KeyType*)(lxch_tmp->ptr());
  bool flag = false;
  for (int i = 0; i < len; i++) {
    if (LxchDebug::get_ins()->lxch_test_set.find(lxch_tmp_1[i]) == LxchDebug::get_ins()->lxch_test_set.end()) {
      VLOG(0) << "lxchdebugaaaaa  " << lxch_tmp_1[i] << "  " << i << "  " << len;
      flag = true;
    }
  }
  if (flag) {
    abort();
  }
  */

  
  if (lxch_run_flag_ == 0 || lxch_run_flag_ == 2) {

  if (len == 0) {
    return;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

  int grid_size = (len - 1) / block_size_ + 1;

  auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  //
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  size_t val_type_size = 0;
  if (!multi_mf_dim_) {
    val_type_size = sizeof(ValType);
  } else {
    val_type_size = g_transfor->get_gpu_value_size(max_mf_dim_);
  }

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::Alloc(place, len * val_type_size);
  ValType* d_shard_vals_ptr = reinterpret_cast<ValType*>(d_shard_vals->ptr());

  split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr, num);

  fill_shard_key<<<grid_size, block_size_, 0, stream>>>(d_shard_keys_ptr,
                                                        d_keys, d_idx_ptr, len);

  cudaMemcpyAsync(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(num, i, shard_len * sizeof(KeyType),
                   shard_len * val_type_size);
  }

  walk_to_dest(num, total_gpu, h_left, h_right, d_shard_keys_ptr, NULL);

  std::vector<platform::Timer> time_lines;
  time_lines.resize(total_gpu);

  for (int i = 0; i < total_gpu; ++i) {
    time_lines[i].Start();
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->rwlock_->RDLock();
      tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                      reinterpret_cast<ValType*>(node.val_storage),
                      h_right[i] - h_left[i] + 1,
                      resource_->remote_stream(i, num));
    } else {
      ptr_tables_[i]->rwlock_->RDLock();
      ptr_tables_[i]->get(resource_->remote_stream(i, num),
                          reinterpret_cast<KeyType*>(node.key_storage),
                          reinterpret_cast<ValType*>(node.val_storage), h_right[i] - h_left[i] + 1);
    }
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    if (!multi_mf_dim_) {
      tables_[i]->rwlock_->UNLock();
    } else {
      ptr_tables_[i]->rwlock_->UNLock();
    }
    time_lines[i].Pause();
  }

  if (!multi_mf_dim_) {
    walk_to_src(num, total_gpu, h_left, h_right, d_shard_vals_ptr);
  } else {
    walk_to_src(num, total_gpu, h_left, h_right, reinterpret_cast<char*>(d_shard_vals_ptr), val_type_size);
  }

  for (int i = 0; i < total_gpu; ++i) {
    auto& node = path_[num][i].nodes_.front();
    cudaStreamSynchronize(node.out_stream);
  }

  if (!multi_mf_dim_) {
    fill_dvals<<<grid_size, block_size_, 0, stream>>>(d_shard_vals_ptr, d_vals,
                                                      d_idx_ptr, len);
  } else {
    dy_mf_fill_dvals<<<grid_size, block_size_, 0, stream>>>(
        d_shard_vals_ptr, d_vals, d_idx_ptr, len, val_type_size);
  }
  cudaStreamSynchronize(stream);
  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(num, i);
  }
  
  } //lxch_run_flag_
  auto lxch_step_2 = platform::Timer::lxch_get_base_time();

  ValType* d_out_put_var = d_vals;
  if (lxch_run_flag_ == 2) {
    size_t val_type_size = g_transfor->get_gpu_value_size(max_mf_dim_);
    cudaMalloc((void**)&d_out_put_var, len * val_type_size);
  }
  
  auto lxch_step_3 = platform::Timer::lxch_get_base_time();
  if (lxch_run_flag_ == 1 || lxch_run_flag_ == 2) {
    lxch_pull_sparse(num, d_keys, d_out_put_var, len);
  }
  auto lxch_step_4 = platform::Timer::lxch_get_base_time();

  VLOG(0) << "lxchdebuga  " << lxch_step_2 - lxch_step_1 << "  "  << lxch_step_4 - lxch_step_3;

  /*
  total_size = 0;
  free_size = 0;
  cudaMemGetInfo(&free_size, &total_size);
  VLOG(0) << "lxch cudamemory2 device_id:" << num << "  " << "  memory:" << free_size / 1024/1024/1204;
  */
  if (lxch_run_flag_ == 2) {
    size_t val_type_size = g_transfor->get_gpu_value_size(max_mf_dim_);
    char* h_src = (char*)malloc(len * val_type_size);
    char* h_src_1 = (char*)malloc(len * val_type_size);
    cudaMemcpy(h_src, d_vals, len * val_type_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_src_1, d_out_put_var, len * val_type_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len * val_type_size; i+=val_type_size) {
      ValType* h_val = (ValType*)(h_src + i);
      ValType* h_val_1 = (ValType*)(h_src_1 + i);
      if (!h_val->is_equal(*h_val_1)) {
        abort();
      }
    }
    cudaFree(d_out_put_var);
  }
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::lxch_push_sparse(int gpu_num,
                                                        KeyType* d_keys,
                                                        GradType* d_grads,
                                                        size_t len,
                                                        Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto& split_info = split_key_info_[gpu_num];
  size_t grad_value_size = g_transfor->get_gpu_push_value_size(max_mf_dim_);

  uint32_t* d_sort_idxs_ptr = (uint32_t*)(split_info.d_sort_idxs->ptr());
  revert_info* d_revert_info_1_ptr = (revert_info*)(split_info.d_revert_info_1->ptr());
  uint32_t* d_uniq_size_1_ptr = (uint32_t*)(split_info.d_uniq_size_1->ptr());
  uint32_t h_uniq_size = *(reinterpret_cast<uint32_t*>(split_info.h_uniq_size->ptr()));

  auto d_shard_grads = memory::Alloc(place, h_uniq_size * grad_value_size);
  GradType* d_shard_grads_ptr = reinterpret_cast<GradType*>(d_shard_grads->ptr());

  GradType* d_src_grads = d_grads;
  if (lxch_run_flag_ == 2) {
    cudaMalloc((void**)&d_src_grads, len * grad_value_size);
    cudaMemcpyAsync(d_src_grads, d_grads, len * grad_value_size, cudaMemcpyDeviceToDevice, stream);
  }
  
  int* normal_size = (int*)split_info.h_normal_size->ptr();
  uint64_t size_1  = normal_size[1] - normal_size[0] > 0 ?  normal_size[1] - normal_size[0] : 0;
  uint64_t size_4  = normal_size[2] - normal_size[1] > 0 ?  normal_size[2] - normal_size[1] : 0;
  uint64_t size_16  = normal_size[3] - normal_size[2] > 0 ?  normal_size[3] - normal_size[2] : 0;
  uint64_t size_64  = normal_size[4] - normal_size[3] > 0 ?  normal_size[4] - normal_size[3] : 0;
  uint64_t size_256 = h_uniq_size - normal_size[4] > 0 ?  h_uniq_size - normal_size[4] : 0;
  uint64_t kernel_1 = (size_1 + 255) / 256 * 256;
  uint64_t kernel_4 = (size_4 * 4 + 255) / 256 * 256;
  uint64_t kernel_16 = (size_16 * 16 + 255) / 256 * 256;
  uint64_t kernel_64 = (size_64 * 64 + 255) / 256 * 256;
  uint64_t kernel_256 = size_256 * 256;
  uint64_t kernel_all = kernel_1 + kernel_4 + kernel_16 + kernel_64 + kernel_256;
  int grid_size = (kernel_all - 1) / block_size_ + 1;
  lxch_merge_gradient_kernel<<<grid_size, block_size_, 0, stream>>>(d_shard_grads_ptr, d_src_grads,
                                  d_sort_idxs_ptr, d_revert_info_1_ptr, d_uniq_size_1_ptr,
                                  h_uniq_size, grad_value_size, 
                                  size_1, size_4, size_16, size_64, size_256, 
                                  kernel_1, kernel_1 + kernel_4, kernel_1 + kernel_4 + kernel_16, 
                                  kernel_1 + kernel_4 + kernel_16 + kernel_64, kernel_all);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  if (lxch_run_flag_ == 2) {
    cudaFree(d_src_grads);
  }
  
  int* h_left = reinterpret_cast<int*>(split_info.h_shard_left->ptr());
  int* h_right = reinterpret_cast<int*>(split_info.h_shard_right->ptr());

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(gpu_num, i, 0,
                     shard_len * grad_value_size);
  }

  walk_to_dest(gpu_num, total_gpu, h_left, h_right, reinterpret_cast<revert_info*>(NULL),
               reinterpret_cast<char*>(d_shard_grads_ptr), grad_value_size);
  
  if (0) {
    //每个线程操作对应卡的更新(做去重)
    //step1 等待传输完成
    for (int i = 0; i < total_gpu; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      auto& node = path_[gpu_num][i].nodes_.back();
      cudaStreamSynchronize(node.in_stream);
    }
    //step2 做一个同步操作, 等待所有的卡都执行到这里
    lxch_barrir();
    auto lxch_step_1 = platform::Timer::lxch_get_base_time();
    //step3 开始进行梯度数据的合并操作
    platform::CUDADeviceGuard guard(dev_id);
    //step3.1 计算一共有多少个key
    size_t total_key_size_len = 0;
    size_t total_value_size_len = 0;
    for (int i = 0; i < total_gpu; ++i) {
      auto& node = path_[i][gpu_num].nodes_.back();
      total_key_size_len += node.key_bytes_len;
      total_value_size_len += node.val_bytes_len;
    }
    size_t total_key_size = total_key_size_len / sizeof(revert_info);
    //step3.2 将数据copy到一块连续的空间上面来
    auto d_src_key_info = memory::Alloc(place, total_key_size_len);
    revert_info* d_src_key_info_ptr = reinterpret_cast<revert_info*>(d_src_key_info->ptr());
    auto d_src_value =  memory::Alloc(place, total_value_size_len);
    GradType* d_src_value_ptr = reinterpret_cast<GradType*>(d_src_value->ptr());
    size_t offset_key = 0;
    size_t offset_value = 0;
    for (int i = 0; i < total_gpu; ++i) {
      auto& node = path_[i][gpu_num].nodes_.back();
      cudaMemcpyAsync(((char*)d_src_key_info_ptr) + offset_key, node.key_storage, node.key_bytes_len, cudaMemcpyDefault, stream);
      cudaMemcpyAsync(((char*)d_src_value_ptr) + offset_value, node.val_storage, node.val_bytes_len, cudaMemcpyDefault, stream);
      offset_key +=  node.key_bytes_len;
      offset_value += node.val_bytes_len;
    }
    //step3.3 填充索引 & key信息
    auto d_src_idxs = memory::Alloc(place, total_key_size * sizeof(uint32_t));
    uint32_t* d_src_idxs_ptr = reinterpret_cast<uint32_t*>(d_src_idxs->ptr());
    int grid_size = (total_key_size - 1) / block_size_ + 1;
    fill_idx<<<grid_size, block_size_, 0, stream>>>(d_src_idxs_ptr, total_key_size);
    auto d_src_key = memory::Alloc(place, total_key_size * sizeof(uint64_t));
    uint64_t* d_src_key_ptr = reinterpret_cast<uint64_t*>(d_src_key->ptr());
    fill_key<<<grid_size, block_size_, 0, stream>>>(d_src_key_ptr, d_src_key_info_ptr, total_key_size);
    //step3.4 按key排序
    auto d_sort_key = memory::Alloc(place, total_key_size * sizeof(uint64_t));
    uint64_t* d_sort_key_ptr = reinterpret_cast<uint64_t*>(d_sort_key->ptr());
    auto d_sort_idx = memory::Alloc(place, total_key_size * sizeof(uint32_t));
    uint32_t* d_sort_idx_ptr = reinterpret_cast<uint32_t*>(d_sort_idx->ptr());
    size_t temp_storage_bytes = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
        NULL, temp_storage_bytes, d_src_key_ptr, d_sort_key_ptr, d_src_idxs_ptr, 
        d_sort_idx_ptr, total_key_size, 0, 8 * sizeof(KeyType), stream));
    auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
        d_temp_storage->ptr(), temp_storage_bytes, d_src_key_ptr, d_sort_key_ptr, d_src_idxs_ptr, 
        d_sort_idx_ptr, total_key_size, 0, 8 * sizeof(KeyType), stream));
    //step3.4 去重
    temp_storage_bytes = 0;
    uint64_t* d_uniq_keys_ptr = d_src_key_ptr;
    uint32_t* d_uniq_key_size_ptr = d_src_idxs_ptr;
    auto d_uniq_size = memory::Alloc(place, sizeof(uint32_t));
    uint32_t* d_uniq_size_ptr = (uint32_t*)(d_uniq_size->ptr());
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
        NULL, temp_storage_bytes, d_sort_key_ptr, d_uniq_keys_ptr, d_uniq_key_size_ptr,
        d_uniq_size_ptr, total_key_size, stream));
    if (d_temp_storage->size() < temp_storage_bytes) {
      d_temp_storage = NULL;
      d_temp_storage = memory::Alloc(place, temp_storage_bytes);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
        d_temp_storage->ptr(), temp_storage_bytes, d_sort_key_ptr, d_uniq_keys_ptr, 
        d_uniq_key_size_ptr, d_uniq_size_ptr, total_key_size, stream));
    auto h_uniq_size_cur = memory::Alloc(phi::GPUPinnedPlace(), sizeof(uint32_t));
    uint32_t* h_uniq_size_cur_ptr = (uint32_t*)(h_uniq_size_cur->ptr());
    cudaMemcpyAsync((void*)h_uniq_size_cur_ptr, d_uniq_size_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    uint32_t uniq_size = *h_uniq_size_cur_ptr;
    //step3.5 做一个前缀规约操作
    auto d_offset = memory::Alloc(place, sizeof(uint32_t) * uniq_size);
    uint32_t* d_offset_ptr = (uint32_t*)(d_offset->ptr());
    temp_storage_bytes = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
        NULL, temp_storage_bytes, d_uniq_key_size_ptr, d_offset_ptr, uniq_size,
        stream));
    if (d_temp_storage->size() < temp_storage_bytes) {
      d_temp_storage = NULL;
      d_temp_storage = memory::Alloc(place, temp_storage_bytes);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
        d_temp_storage->ptr(), temp_storage_bytes, d_uniq_key_size_ptr, d_offset_ptr,
        uniq_size, stream));
    //step3.6 调用kernel进行梯度的合并操作了
    auto d_merge_grads = memory::Alloc(place, uniq_size * grad_value_size);
    GradType* d_merge_grads_ptr = reinterpret_cast<GradType*>(d_merge_grads->ptr());
    grid_size = (uniq_size - 1) / block_size_ + 1;
    merge_gradient_kernel<<<grid_size, block_size_, 0, stream>>>(d_offset_ptr, d_uniq_key_size_ptr, 
                d_sort_idx_ptr, d_src_value_ptr, d_merge_grads_ptr, uniq_size, grad_value_size);
    //step3.7 调用kernel进行key_info的重整
    auto d_merge_keyinfo = memory::Alloc(place, uniq_size * sizeof(revert_info));
    revert_info* d_merge_keyinfo_ptr = reinterpret_cast<revert_info*>(d_merge_keyinfo->ptr());
    merge_key_info_kernel<<<grid_size, block_size_, 0, stream>>>(d_offset_ptr, d_uniq_key_size_ptr, 
                d_sort_idx_ptr, d_src_key_info_ptr, d_merge_keyinfo_ptr, uniq_size);
    //step4 梯度更新操作了
    ptr_tables_[gpu_num]->rwlock_->WRLock();
    ptr_tables_[gpu_num]->update(d_merge_keyinfo_ptr,
                           d_merge_grads_ptr, uniq_size,
                           stream, sgd, dev_id);
    ptr_tables_[gpu_num]->rwlock_->UNLock();
    lxch_barrir();
    auto lxch_step_2 = platform::Timer::lxch_get_base_time();
    VLOG(0) << "lxchdebugc  " << lxch_step_2 - lxch_step_1;
  } else if (1) {
    for (int i = 0; i < total_gpu; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      auto& node = path_[gpu_num][i].nodes_.back();
      cudaStreamSynchronize(node.in_stream);
    }
    lxch_barrir();
    auto lxch_step_1 = platform::Timer::lxch_get_base_time();
    platform::CUDADeviceGuard guard(resource_->dev_id(gpu_num));
    auto s_stream = resource_->remote_stream(gpu_num, gpu_num);
    ptr_tables_[gpu_num]->rwlock_->WRLock();
    for (int i = 0; i < total_gpu; ++i) {
      auto& other_split_info = split_key_info_[i];
      int* other_h_left = reinterpret_cast<int*>(other_split_info.h_shard_left->ptr());
      int* other_h_right = reinterpret_cast<int*>(other_split_info.h_shard_right->ptr());
      if (other_h_left[gpu_num] == -1 || other_h_right[gpu_num] == -1) {
        continue;
      }
      auto& node = path_[i][gpu_num].nodes_.back();
      {
        ptr_tables_[gpu_num]->update(reinterpret_cast<revert_info*>(node.key_storage),
                               reinterpret_cast<GradType*>(node.val_storage), other_h_right[gpu_num] - other_h_left[gpu_num] + 1,
                               s_stream, sgd, dev_id);
      }
    }
    cudaStreamSynchronize(s_stream);
    ptr_tables_[gpu_num]->rwlock_->UNLock();
    lxch_barrir();
    auto lxch_step_2 = platform::Timer::lxch_get_base_time();
    VLOG(0) << "lxchdebugc  " << lxch_step_2 - lxch_step_1;
  } else {
    //每个线程操作多张卡的更新
    lxch_barrir();
    auto lxch_step_1 = platform::Timer::lxch_get_base_time();
    std::vector<platform::Timer> time_lines;
    time_lines.resize(total_gpu);

    for (int i = 0; i < total_gpu; ++i) {
      time_lines[i].Start();
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      auto& node = path_[gpu_num][i].nodes_.back();
      cudaStreamSynchronize(node.in_stream);

      platform::CUDADeviceGuard guard(resource_->dev_id(i));
      {
        ptr_tables_[i]->rwlock_->WRLock();
        ptr_tables_[i]->update(reinterpret_cast<revert_info*>(node.key_storage),
                               reinterpret_cast<GradType*>(node.val_storage), h_right[i] - h_left[i] + 1,
                               resource_->remote_stream(i, gpu_num), sgd, dev_id);
      
      }
    }
    for (int i = 0; i < total_gpu; ++i) {
      cudaStreamSynchronize(resource_->remote_stream(i, gpu_num));
      if (h_left[i] != -1) {
        if (!multi_mf_dim_) {
          tables_[i]->rwlock_->UNLock();
        } else {
          ptr_tables_[i]->rwlock_->UNLock();
        }
        time_lines[i].Pause();
      }
    }
    auto lxch_step_2 = platform::Timer::lxch_get_base_time();
    VLOG(0) << "lxchdebugc  " << lxch_step_2 - lxch_step_1;
    lxch_barrir();
  }

  if (lxch_run_flag_ == 2) {
    /*
    uint32_t* h_sort_idxs = new uint32_t[len];
    revert_info* h_revert_info_1 = new revert_info[h_uniq_size];
    uint32_t* h_uniq_key_size = new uint32_t[len];
    uint32_t* h_uniq_size_1 = new uint32_t[h_uniq_size];
    cudaMemcpy(h_sort_idxs, d_sort_idxs_ptr, len*sizeof(uint32_t), cudaMemcpyDefault);
    cudaMemcpy(h_revert_info_1, d_revert_info_1_ptr, h_uniq_size*sizeof(revert_info), cudaMemcpyDefault);
    cudaMemcpy(h_uniq_size_1, d_uniq_size_1_ptr, h_uniq_size*sizeof(uint32_t), cudaMemcpyDefault);
    */

    revert_info* key_info = new revert_info[h_uniq_size];
    char* value_info =  new char[grad_value_size*h_uniq_size];
    size_t cur_offset = 0;
    for (int i = 0; i < total_gpu; ++i) {
      platform::CUDADeviceGuard guard(resource_->dev_id(i));
      auto& node = path_[gpu_num][i].nodes_.back();
      size_t shard_len = h_right[i] - h_left[i] + 1;
      cudaMemcpy(key_info + cur_offset, node.key_storage, shard_len* sizeof(revert_info), cudaMemcpyDefault);
      cudaMemcpy(value_info + cur_offset * grad_value_size, node.val_storage, shard_len * grad_value_size, cudaMemcpyDefault);
      cur_offset += shard_len;
    }
    if (cur_offset != h_uniq_size) {
      abort();
    }
    if (1) {
      //test key is shard right
      cur_offset = 0;
      for (int i = 0; i < total_gpu; ++i) {
        size_t shard_len = h_right[i] - h_left[i] + 1;
        for (size_t j = cur_offset; j < cur_offset + shard_len; j++) {
          if (key_info[j].key % total_gpu != i) {
            abort();
          }
        }
        cur_offset += shard_len;
      }

      //test_2 merge_grad 是否正确
      std::map<uint64_t, char*> new_map;
      for (uint32_t i = 0; i < h_uniq_size; i++) {
        if (new_map.find(key_info[i].key) != new_map.end()) {
          abort();
        }
        new_map[key_info[i].key] = value_info + i * grad_value_size;
      }
      uint64_t* src_key = new uint64_t[len];
      char* src_value = new char[len * grad_value_size];
      cudaMemcpy(src_key, d_keys, len * sizeof(uint64_t), cudaMemcpyDefault);
      cudaMemcpy(src_value, d_grads, len * grad_value_size, cudaMemcpyDefault);

      std::map<uint64_t, char*> old_map;
      for (size_t i = 0; i < len; i++) {
        uint64_t cur_key = src_key[i];
        char* cur_value = src_value + i * grad_value_size;
        if (old_map.find(cur_key) == old_map.end()) {
          old_map[cur_key] = cur_value;
        } else {
          GradType* des = (GradType*) (old_map[cur_key]);
          GradType* src = (GradType*) cur_value;
          *des += *src;
        }
      }

      if (old_map.size() != new_map.size()) {
        abort();
      }
      for (auto iter = new_map.begin(); iter != new_map.end(); iter++) {
        uint64_t new_key = iter->first;
        GradType* new_value = (GradType*)iter->second;
        if (old_map.find(new_key)  == old_map.end()) {
          abort();
        }
        GradType* old_value = (GradType*)old_map[new_key];
        if (!old_value->is_equal(*new_value)) {
          abort();
        }
      }
      delete []src_key;
      delete []src_value;
    }
    delete []key_info;
    delete []value_info;
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(gpu_num, i);
  }
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::push_sparse(int gpu_num,
                                                        KeyType* d_keys,
                                                        GradType* d_grads,
                                                        size_t len,
                                                        Sgd& sgd) {  // NOLINT
  /*
  size_t total_size = 0;
  size_t free_size = 0;
  cudaMemGetInfo(&free_size, &total_size);
  VLOG(0) << "lxch cudamemory3 device_id:" << gpu_num << "  " << "  memory:" << free_size / 1024/1024/1204;
  */
  lxch_barrir();
  auto lxch_step_3 = platform::Timer::lxch_get_base_time();
  if (lxch_run_flag_ == 1 || lxch_run_flag_ == 2) {
    lxch_push_sparse(gpu_num, d_keys, d_grads, len, sgd);
  }
  auto lxch_step_4 = platform::Timer::lxch_get_base_time();

  /*
  total_size = 0;
  free_size = 0;
  cudaMemGetInfo(&free_size, &total_size);
  VLOG(0) << "lxch cudamemory4 device_id:" << gpu_num << "  " << "  memory:" << free_size / 1024/1024/1204;
  */
  if (len == 0) {
    return;
  }
  auto lxch_step_1 = platform::Timer::lxch_get_base_time();
  if (lxch_run_flag_ == 0) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t grad_value_size = g_transfor->get_gpu_push_value_size(max_mf_dim_);

  // int h_left[total_gpu];   // NOLINT
  // int h_right[total_gpu];  // NOLINT
  auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());

  auto d_shard_grads = memory::Alloc(place, len * grad_value_size);
  GradType* d_shard_grads_ptr = reinterpret_cast<GradType*>(d_shard_grads->ptr());

  cudaStreamSynchronize(stream);
  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, NULL, len, uniq_len);

  int grid_size = (uniq_len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr,
                       gpu_num);

  if (!multi_mf_dim_) {
    fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
        d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
        uniq_len);
  } else {
    dy_mf_fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
        d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
        uniq_len, grad_value_size);
  }

  cudaMemcpyAsync(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    if (!multi_mf_dim_) {
      create_storage(gpu_num, i, shard_len * sizeof(KeyType),
                     shard_len * sizeof(GradType));
    } else {
      create_storage(gpu_num, i, shard_len * sizeof(KeyType),
                     shard_len * grad_value_size);
    }
  }

  if (!multi_mf_dim_) {
    walk_to_dest(gpu_num, total_gpu, h_left, h_right, d_shard_keys_ptr,
               d_shard_grads_ptr);
  } else {
    walk_to_dest(gpu_num, total_gpu, h_left, h_right, d_shard_keys_ptr,
               reinterpret_cast<char*>(d_shard_grads_ptr), grad_value_size);
  }

  std::vector<platform::Timer> time_lines;
  time_lines.resize(total_gpu);

  for (int i = 0; i < total_gpu; ++i) {
    time_lines[i].Start();
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[gpu_num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);

    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->rwlock_->WRLock();
      tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                         reinterpret_cast<GradType*>(node.val_storage),
                         h_right[i] - h_left[i] + 1, sgd,
                         resource_->remote_stream(i, gpu_num));
    } else {
      ptr_tables_[i]->rwlock_->WRLock();
      ptr_tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                             reinterpret_cast<GradType*>(node.val_storage), h_right[i] - h_left[i] + 1,
                             resource_->remote_stream(i, gpu_num), sgd);
    }
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, gpu_num));
    if (h_left[i] != -1) {
      if (!multi_mf_dim_) {
        tables_[i]->rwlock_->UNLock();
      } else {
        ptr_tables_[i]->rwlock_->UNLock();
      }
      time_lines[i].Pause();
    }
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(gpu_num, i);
  }
  }
  auto lxch_step_2 = platform::Timer::lxch_get_base_time();

  VLOG(0) << "lxchdebugb  " << lxch_step_2 - lxch_step_1 << "  "
                            << lxch_step_4 - lxch_step_3 << "  ";

}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::update_one_table(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDADeviceGuard guard(dev_id);
  tables_[gpu_num]->rwlock_->WRLock();
  tables_[gpu_num]->update(d_keys, d_grads, len, sgd,
                           resource_->remote_stream(gpu_num, gpu_num));
  tables_[gpu_num]->rwlock_->UNLock();
  cudaStreamSynchronize(resource_->remote_stream(gpu_num, gpu_num));
}

template <typename KeyType, typename ValType, typename GradType>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType>::push_sparse_multi_node(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, len, uniq_len);

  uniq_len = gather_one_node_grad(gpu_num, d_keys, d_grads, uniq_len);

  uniq_len = gather_multi_node_grad(gpu_num, storage_[gpu_num].local_keys,
                                    storage_[gpu_num].local_grads, uniq_len);

  update_one_table(gpu_num, storage_[gpu_num].local_keys,
                   storage_[gpu_num].local_grads, uniq_len, sgd);
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::gather_one_node_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;

  ncclComm_t nccl_inner_comm = nccl_inner_comms_[gpu_num];
  // alloc for size
  int h_node_len[total_gpu];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[gpu_num] = len;

  cudaMemcpy(d_node_len + gpu_num, h_node_len + gpu_num, sizeof(int),
             cudaMemcpyHostToDevice);

  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::ncclAllGather((const void*)(d_node_len + gpu_num),
                                       (void*)d_node_len, 1, ncclInt,  // NOLINT
                                       nccl_inner_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * total_gpu,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  storage.alloc(max_size * total_gpu);

  // allgather keys and grads
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_keys, storage.all_keys, max_size, ncclUint64, nccl_inner_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_grads, storage.all_grads, max_size * sizeof(GradType), ncclUint8,
      nccl_inner_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT
  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  int merge_num = 0;
  for (int i = 0; i < total_gpu; ++i) {
    int index = i * max_size;
    auto d_idx = memory::Alloc(place, h_node_len[i] * sizeof(int));
    int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

    cudaMemset(d_left_ptr, -1, total_gpu * sizeof(int));
    cudaMemset(d_right_ptr, -1, total_gpu * sizeof(int));

    split_input_to_shard(storage.all_keys + index, d_idx_ptr, h_node_len[i],
                         d_left_ptr, d_right_ptr, gpu_num);
    cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
               cudaMemcpyDeviceToHost);

    int grid_size = (h_node_len[i] - 1) / block_size_ + 1;
    fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
        storage.local_keys + merge_num, storage.all_keys + index,
        storage.local_grads + merge_num, storage.all_grads + index,
        d_idx_ptr + h_left[gpu_num], h_right[gpu_num] - h_left[gpu_num] + 1);
    merge_num = merge_num + h_right[gpu_num] - h_left[gpu_num] + 1;
  }

  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, merge_num, ret);
  return ret;
}

template <typename KeyType, typename ValType, typename GradType>
int HeterComm<KeyType, ValType, GradType>::gather_multi_node_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;
  ncclComm_t nccl_inter_comm = nccl_inter_comms_[gpu_num];
  // alloc for size
  int h_node_len[node_size_];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[0] = len;

  cudaMemcpy(d_node_len, h_node_len, sizeof(int), cudaMemcpyHostToDevice);

  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_node_len, d_node_len, 1, ncclInt, nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  storage.alloc(max_size * node_size_);

  // allgather keys and grads
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_keys, storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_grads, storage.all_grads, max_size * sizeof(GradType), ncclUint8,
      nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  int merge_num = 0;
  for (int i = 0; i < node_size_; ++i) {
    int index = i * max_size;
    cudaMemcpyAsync(storage.local_keys + merge_num, storage.all_keys + index,
                    h_node_len[i], cudaMemcpyDefault, stream);
    cudaMemcpyAsync(storage.local_grads + merge_num, storage.all_grads + index,
                    h_node_len[i], cudaMemcpyDefault, stream);
    merge_num += h_node_len[i];
  }

  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, merge_num, ret);
  return ret;
}

template <typename KeyType, typename ValType, typename GradType>
void HeterComm<KeyType, ValType, GradType>::end_pass() {
  int total_gpu = resource_->total_gpu();
  std::vector<std::thread> threads;

  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    platform::CUDADeviceGuard guard(dev_id);
    tables_[index]->dump_to_cpu(dev_id, stream);
  };

  if (!multi_mf_dim_) {
    for (int i = 0; i < total_gpu; ++i) {
      threads.push_back(std::thread(dump_to_cpu_func, i));
    }
    for (auto& t : threads) {
      t.join();
    }
  }
}

// template <typename KeyType, typename ValType, typename GradType>
// void HeterComm<KeyType, ValType, GradType>::dump_to_cpu(int index) {
//  auto stream = resource_->local_stream(index, 0);
//  int dev_id = resource_->dev_id(index);
//  platform::CUDADeviceGuard guard(dev_id);
//  tables_[index]->dump_to_cpu(dev_id, stream);
//}

}  // end namespace framework
}  // end namespace paddle
#endif
