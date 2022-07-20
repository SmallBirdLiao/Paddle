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

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/framework/heter_util.h"

namespace paddle {
namespace framework {

template <typename value_type>
struct ReplaceOp {
  __host__ __device__ value_type operator()(value_type new_value,
                                            value_type old_value) {
    return new_value;
  }
};

template <typename Table>
__global__ void insert_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals,
                              size_t len) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    kv.first = keys[i];
    kv.second = vals[i];
    auto it = table->insert(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table>
__global__ void insert_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              size_t len, char* pool, size_t feature_value_size,
                              int start_index) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
    kv.first = keys[i];
    uint64_t offset = uint64_t(start_index + i) * feature_value_size;
    kv.second = (Table::mapped_type)(pool + offset);
    auto it = table->insert(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table>
__global__ void search_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              typename Table::mapped_type* const vals,
                              size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      vals[i] = it->second;
    } else {
      printf("apull miss key: %llu", keys[i]);
    }
  }
}

template <typename Table, typename ValType>
__global__ void dy_mf_search_kernel(Table* table,
                                    const typename Table::key_type* const keys,
                                    ValType* vals, size_t len,
                                    size_t pull_feature_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    char* d_value = (char*)(vals);
    if (it != table->end()) {
      uint64_t offset = i * pull_feature_value_size;
      ValType* cur = (ValType*)(d_value + offset);
      ValType& input = *(ValType*)(it->second);
      *cur = input;
    } else {
      if (keys[i] != 0) printf("bpull miss key: %llu", keys[i]);
      ValType* cur = (ValType*)(d_value + i * pull_feature_value_size);
      *cur = ValType();
    }
  }
}

template <typename Table, typename ValType>
__global__ void lxch_dy_mf_search_kernel(Table* table,
                                   revert_info* keys,
                                    ValType* vals, size_t len,
                                    size_t pull_feature_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i].key);
    char* d_value = (char*)(vals);
    if (it != table->end()) {
      uint64_t offset = i * pull_feature_value_size;
      ValType* cur = (ValType*)(d_value + offset);
      ValType& input = *(ValType*)(it->second);
      *cur = input;
      keys[i].value_ptr = (void*)(it->second);
    } else {
      if (keys[i].key != 0) printf("cpull miss key: %llu", keys[i].key);
      ValType* cur = (ValType*)(d_value + i * pull_feature_value_size);
      *cur = ValType();
      keys[i].value_ptr = nullptr;
    }
  }
}

__global__ void curand_init_kernel(curandState* p_value, int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    curand_init(clock64(), i, 0, p_value + i);
  }
}

class CuRandState {
 public:
  CuRandState() = default;
  CuRandState(const CuRandState&) = delete;
  CuRandState(CuRandState&&) = delete;
  ~CuRandState() { CHECK(cudaFree(states_) == cudaSuccess); }
  curandState* get(size_t size, gpuStream_t stream) {
    if (size > size_) {
      size_t new_size = size * 2;
      curandState* new_state = nullptr;
      CHECK(cudaMalloc(reinterpret_cast<void**>(&new_state),
                       new_size * sizeof(curandState)) == cudaSuccess);
      if (size_ > 0) {
        CHECK(cudaMemcpyAsync(new_state, states_, size_ * sizeof(curandState),
                              cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
      }
      int init_len = new_size - size_;
      const int BLOCK_SIZE_{256};
      const int init_kernel_grid = (init_len - 1) / BLOCK_SIZE_ + 1;
      curand_init_kernel<<<init_kernel_grid, BLOCK_SIZE_, 0, stream>>>(
          new_state + size_, init_len);
      if (size_ != 0) {
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(cudaFree(states_) == cudaSuccess);
      }
      states_ = new_state;
      size_ = new_size;
    }
    return states_;
  }

  static HeterObjectPool<CuRandState>& pool() {
    static HeterObjectPool<CuRandState> p;
    return p;
  }
  static HeterObjectPool<CuRandState>& new_pool(int dev_id) {
    static HeterObjectPool<CuRandState> p[100];
    return p[dev_id];
  }

  static std::shared_ptr<CuRandState> get() { return pool().Get(); }
  static std::shared_ptr<CuRandState> new_get(int dev_id) { return new_pool(dev_id).Get(); }

  static void CUDART_CB pushback_cu_rand_state(void* data) {
    auto state = static_cast<std::shared_ptr<CuRandState>*>(data);
    pool().Push(std::move(*state));
    delete state;
  }

  static void push(std::shared_ptr<CuRandState> state, gpuStream_t stream) {
    CHECK(cudaLaunchHostFunc(stream, pushback_cu_rand_state,
                             new std::shared_ptr<CuRandState>(
                                 std::move(state))) == cudaSuccess);
  }
  static void new_push(std::shared_ptr<CuRandState> state, int dev_id) {
    new_pool(dev_id).Push(state);
  }

 private:
  size_t size_ = 0;
  curandState* states_ = nullptr;
};

template <typename Table, typename GradType, typename Sgd>
__global__ void update_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const GradType* const grads, curandState* p_state,
                              size_t len, Sgd sgd) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      sgd.update_value((it.getter())->second, grads[i], p_state[i]);
    } else {
      printf("push miss key: %llu", keys[i]);
    }
  }
}

template <typename Table, typename GradType, typename Sgd>
__global__ void update_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const GradType* const grads, size_t len,
                              Sgd sgd) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      sgd.update_value((it.getter())->second, grads[i]);
    } else {
      printf("push miss key: %llu", keys[i]);
    }
  }
}

template <typename Table, typename GradType, typename Sgd>
__global__ void dy_mf_update_kernel(Table* table,
                                    const typename Table::key_type* const keys,
                                    const GradType* grads, size_t len,
                                    Sgd sgd, size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      char* grads_tmp = (char*)(grads);
      GradType* cur = (GradType*)(grads_tmp + i * grad_value_size);
      sgd.dy_mf_update_value((it.getter())->second, *cur);
    } else {
      if (keys[i] != 0) printf("push miss key: %llu", keys[i]);
    }
  }
}

template <typename Table, typename GradType, typename Sgd>
__global__ void dy_mf_update_kernel(Table* table,
                                    const typename Table::key_type* const keys,
                                    const GradType* grads, curandState* p_state, size_t len,
                                    Sgd sgd, size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      char* grads_tmp = (char*)(grads);
      GradType* cur = (GradType*)(grads_tmp + i * grad_value_size);
      sgd.dy_mf_update_value((it.getter())->second, *cur, p_state[i]);
    } else {
      if(keys[i] != 0) printf("push miss key: %llu", keys[i]);
    }
  }
}

template <typename Table, typename GradType, typename Sgd>
__global__ void lxch_dy_mf_update_kernel(Table* table,
                                    revert_info* const keys,
                                    const GradType* grads, curandState* p_state, size_t len,
                                    Sgd sgd, size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    if (keys[i].value_ptr != nullptr) {
      char* grads_tmp = (char*)(grads);
      GradType* cur = (GradType*)(grads_tmp + i * grad_value_size);
      sgd.dy_mf_update_value((typename Table::mapped_type)keys[i].value_ptr, *cur, p_state[i]);
    } else {
      if(keys[i].key != 0) printf("push miss key: %llu", keys[i]);
    }
  }
}

__global__ void lxch_copy_from_value(revert_info* keys,
                                    char* des_value, size_t len, size_t value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    size_t index = i / value_size;
    size_t offset = i % value_size;
    char* value_ptr = (char*)(keys[index].value_ptr);
    if (value_ptr != nullptr) {
      des_value[i] = value_ptr[offset];
    }
  }
}

__global__ void lxch_copy_to_value(revert_info* keys,
                                    char* des_value, size_t len, size_t value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    size_t index = i / value_size;
    size_t offset = i % value_size;
    char* value_ptr = (char*)(keys[index].value_ptr);
    if (value_ptr != nullptr) {
      value_ptr[offset] = des_value[i];
    }
  }
}


template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity) {
  container_ = new TableContainer<KeyType, ValType>(capacity);
  rwlock_.reset(new phi::RWLock);
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::~HashTable() {
  delete container_;
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::show() {
  container_->print();
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys, ValType* d_vals,
                                      size_t len, gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys,
                                                       d_vals, len);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get(gpuStream_t stream, const KeyType* d_keys, ValType d_vals, size_t len) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  dy_mf_search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len, pull_feature_value_size_);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get(gpuStream_t stream, revert_info* d_keys, ValType d_vals, size_t len) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  lxch_dy_mf_search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len, pull_feature_value_size_);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys,
                                         const ValType* d_vals, size_t len,
                                         gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys,
                                                       d_vals, len);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys, size_t len,
                                         char* pool, size_t feature_value_size,
                                         size_t start_index,
                                         gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  if (pool == NULL) {
    return;
  }
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, len, pool, feature_value_size, start_index);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::dump_to_cpu(int devid, cudaStream_t stream) {
  container_->prefetch(cudaCpuDeviceId, stream);
  std::vector<std::thread> threads;
  size_t num = container_->size();
  KeyType unuse_key = std::numeric_limits<KeyType>::max();
  thrust::pair<KeyType, ValType>* kv = container_->data();

  int thread_num = 8;
  int len_per_thread = num / thread_num;
  int remain = num % thread_num;
  int begin = 0;

  auto dump_func = [unuse_key, kv](int left, int right) {
    for (int i = left; i < right; i++) {
      if (kv[i].first == unuse_key) {
        continue;
      }
      ValType& gpu_val = kv[i].second;
#ifdef PADDLE_WITH_PSLIB
      auto* downpour_value =
          (paddle::ps::DownpourFixedFeatureValue*)(gpu_val.cpu_ptr);
      int downpour_value_size = downpour_value->size();
      if (gpu_val.mf_size > 0 && downpour_value_size == 7) {
        downpour_value->resize(gpu_val.mf_size + downpour_value_size);
      }
      float* cpu_val = downpour_value->data();
      // cpu_val[0] = 0;
      cpu_val[1] = gpu_val.delta_score;
      cpu_val[2] = gpu_val.show;
      cpu_val[3] = gpu_val.clk;
      cpu_val[4] = gpu_val.lr;
      cpu_val[5] = gpu_val.lr_g2sum;
      // useless
      if (cpu_val[6] <= 0) {
        cpu_val[6] = gpu_val.slot * -1;
      } else {
        cpu_val[6] = gpu_val.slot;
      }
      if (gpu_val.mf_size > 0) {
        for (int x = 0; x < gpu_val.mf_size; x++) {
          cpu_val[x + 7] = gpu_val.mf[x];
        }
      }
#endif
#ifdef PADDLE_WITH_PSCORE
      auto* downpour_value =
          (paddle::distributed::FixedFeatureValue*)(gpu_val.cpu_ptr);
      int downpour_value_size = downpour_value->size();
      if (gpu_val.mf_size > 0 && downpour_value_size == 7) {
        downpour_value->resize(gpu_val.mf_size + downpour_value_size);
      }
      float* cpu_val = downpour_value->data();
      // cpu_val[0] = 0;
      cpu_val[2] = gpu_val.delta_score;
      cpu_val[3] = gpu_val.show;
      cpu_val[4] = gpu_val.clk;
      cpu_val[5] = gpu_val.lr;
      cpu_val[6] = gpu_val.lr_g2sum;
      cpu_val[0] = gpu_val.slot;
      if (gpu_val.mf_size > 0) {
        for (int x = 0; x < gpu_val.mf_size; x++) {
          cpu_val[x + 7] = gpu_val.mf[x];
        }
      }
#endif
    }
  };

  for (int i = 0; i < thread_num; i++) {
    threads.push_back(std::thread(
        dump_func, begin, begin + len_per_thread + (i < remain ? 1 : 0)));
    begin += len_per_thread + (i < remain ? 1 : 0);
  }
  for (std::thread& t : threads) {
    t.join();
  }

  // container_->prefetch(devid, stream);
}

template <typename KeyType, typename ValType>
template <typename GradType, typename Sgd>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const GradType* d_grads, size_t len,
                                         Sgd sgd, gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  auto state = CuRandState::get();
  auto d_state = state->get(len, stream);
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_grads, d_state, len, sgd);
  CuRandState::push(state, stream);
}

template <typename KeyType, typename ValType>
template <typename GradType, typename Sgd>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const GradType* d_grads, size_t len,
                                         gpuStream_t stream, Sgd sgd) {
  if (len == 0) {
    return;
  }
  auto state = CuRandState::get();
  auto d_state = state->get(len, stream);
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  dy_mf_update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_grads, d_state, len, sgd, push_grad_value_size_);
  CuRandState::push(state, stream);
}

template <typename KeyType, typename ValType>
template <typename GradType, typename Sgd>
void HashTable<KeyType, ValType>::update(revert_info* d_keys,
                                         const GradType* d_grads, size_t len,
                                         gpuStream_t stream, Sgd sgd, int dev_id) {
  if (len == 0) {
    return;
  }
  if (1) {
    auto state = CuRandState::new_get(dev_id);
    auto d_state = state->get(len, stream);
    int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    lxch_dy_mf_update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
        container_, d_keys, d_grads, d_state, len, sgd, push_grad_value_size_);
    cudaStreamSynchronize(stream);
    CuRandState::new_push(state, dev_id);
  } else {
    auto state = CuRandState::get();
    auto d_state = state->get(len, stream);
    platform::CUDAPlace place = platform::CUDAPlace(dev_id);
    auto d_value = memory::Alloc(place, len * pull_feature_value_size_);
    char* d_value_ptr = (char*)(d_value->ptr());
    int grid_size = (len * pull_feature_value_size_ - 1) / BLOCK_SIZE_ + 1;
    lxch_copy_from_value<<<grid_size, BLOCK_SIZE_, 0, stream>>>(d_keys, d_value_ptr, len * pull_feature_value_size_, pull_feature_value_size_);
    lxch_copy_to_value<<<grid_size, BLOCK_SIZE_, 0, stream>>>(d_keys, d_value_ptr, len * pull_feature_value_size_, pull_feature_value_size_);
    CuRandState::push(state, stream);
    cudaStreamSynchronize(stream);
  }
}

}  // end namespace framework
}  // end namespace paddle
#endif
