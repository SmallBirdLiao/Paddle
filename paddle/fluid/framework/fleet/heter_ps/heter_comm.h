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
#include <thread>
#include <vector>
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "hashtable.h"       // NOLINT
#include "heter_resource.h"  // NOLINT
#include "paddle/fluid/framework/fleet/heter_ps/mem_pool.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/place.h"
#include "thrust/pair.h"
#include "paddle/fluid/platform/timer.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

#define TYPEALIGN(ALIGNVAL, LEN) \
  (((uint64_t)(LEN) + ((ALIGNVAL)-1)) & ~((uint64_t)((ALIGNVAL)-1)))

struct CustomGradMerger {
  template <typename T>
  CUB_RUNTIME_FUNCTION __forceinline__ __device__ T
  operator()(const T& a, const T& b) const {
    T out;
    out.slot = a.slot;
    out.mf_dim = a.mf_dim;
    out.show = a.show + b.show;
    out.clk = a.clk + b.clk;
    out.lr_g = a.lr_g + b.lr_g;
    return out;
  }

  template <typename T>
  __device__ __forceinline__
  void copy_basic_field(T& output, const T& input) {
      output.slot    = input.slot;
      output.show    = input.show;
      output.clk     = input.clk;
      output.mf_dim = input.mf_dim;
      output.lr_g = input.lr_g;
      for(int i = 0; i < output.mf_dim ; ++i) {
         output.mf_g[i] = input.mf_g[i];
      }
  }
  template <typename T>
  __device__ __forceinline__
  void add_basic_field(T& output, const T& input) {
      output.show    += input.show;
      output.clk     += input.clk;
      output.lr_g += input.lr_g;
      for(int i = 0; i < input.mf_dim; ++i) {
         output.mf_g[i] += input.mf_g[i];
      }
  }
};

template <typename KeyType, typename ValType, typename GradType>
class HeterComm {
 public:
  HeterComm(size_t capacity, std::shared_ptr<HeterPsResource> resource);
  virtual ~HeterComm();
  HeterComm(const HeterComm&) = delete;
  HeterComm& operator=(const HeterComm&) = delete;

  void split_input_to_shard(KeyType* d_keys, int* d_idx_ptr, size_t len,
                            int* left, int* right, int gpu_num);
  void lxch_split_input_to_shard(revert_info* d_keys, int* d_idx_ptr, size_t len,
                            int* left, int* right, int gpu_num);
  void merge_grad(int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
                  int& uniq_len);  // NOLINT
  void merge_grad(int gpu_num, KeyType* d_keys, GradType* d_grads, float* mf,
                  size_t len, int& uniq_len);
  void pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void lxch_pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len,
                size_t chunk_size, int stream_num);
  void build_ps(int num, KeyType* h_keys, char* pool, size_t len, size_t feature_value_size,
                size_t chunk_size, int stream_num);
  void dump();
  void show_one_table(int gpu_num);
  int get_index_by_devid(int devid);

  template <typename Sgd>
  void lxch_push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                   Sgd& sgd);  // NOLINT
  
  template <typename Sgd>
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                   Sgd& sgd);  // NOLINT

  template <typename Sgd>
  void push_sparse_multi_node(int num, KeyType* d_keys, GradType* d_grads,
                              size_t len, Sgd& sgd);  // NOLINT

  template <typename Sgd>
  void update_one_table(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                        Sgd& sgd);  // NOLINT

  int gather_one_node_grad(int num, KeyType* d_keys, GradType* d_grads,
                           int len);

  int gather_multi_node_grad(int num, KeyType* d_keys, GradType* d_grads,
                             int len);

  int log2i(int x);

  void set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                              const std::vector<ncclComm_t>& inter_comms,
                              int comm_size) {
    nccl_inner_comms_ = inner_comms;
    nccl_inter_comms_ = inter_comms;
    node_size_ = comm_size;
  }
  
  void set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) {
    
    multi_mf_dim_ = multi_mf_dim;
    max_mf_dim_ = max_mf_dim;
    VLOG(3) << "heter comm set multi multi_mf_dim_: " << multi_mf_dim_ << " max_mf_dim_: " << max_mf_dim_;
  }

  bool need_transfer(int send_id, int receive_id) {
    return ((send_id / 4 != receive_id / 4) && (send_id + 4) % 8 != receive_id);
  }

  // void dump_to_cpu(int index);

  void end_pass();

  int get_transfer_devid(int send_id) { return (send_id + 4) % 8; }

  struct Node {
    Node() {
      key_storage = nullptr;
      val_storage = nullptr;
//      d_key_storage = NULL;
//      d_val_storage = NULL;
    }
    cudaStream_t in_stream;
    cudaStream_t out_stream;
//    memory::allocation::AllocationPtr d_key_storage;
//    memory::allocation::AllocationPtr d_val_storage;
    char* key_storage;
    char* val_storage;
    int sync;
    size_t key_bytes_len;
    size_t val_bytes_len;
    int gpu_num;
  };

  struct Path {
    std::vector<Node> nodes_;
  };

  struct CopyTask {
    Path* path;
    int step;
    CopyTask(Path* path_, int step_) : path(path_), step(step_) {}
  };

  struct LocalStorage {
    LocalStorage() {}
    void init(int size, int dev_id) {
      place_ = platform::CUDAPlace(dev_id);
      alloc(size, true);
    }

    void alloc(int size, bool force = false) {
      if (force || size > all_keys_mem->size()) {
        all_keys_mem.reset();
        all_grads_mem.reset();
        all_keys_mem = memory::Alloc(place_, size * sizeof(KeyType));
        all_grads_mem = memory::Alloc(place_, size * sizeof(GradType));
        all_keys = reinterpret_cast<KeyType*>(all_keys_mem->ptr());
        all_grads = reinterpret_cast<GradType*>(all_grads_mem->ptr());
      }
      if (force || size > local_keys_mem->size()) {
        local_keys_mem.reset();
        local_grads_mem.reset();
        local_keys_mem = memory::Alloc(place_, size * sizeof(KeyType));
        local_grads_mem = memory::Alloc(place_, size * sizeof(GradType));
        local_keys = reinterpret_cast<KeyType*>(local_keys_mem->ptr());
        local_grads = reinterpret_cast<GradType*>(local_grads_mem->ptr());
      }
    }

    platform::CUDAPlace place_;
    std::shared_ptr<memory::Allocation> all_keys_mem;
    std::shared_ptr<memory::Allocation> all_grads_mem;
    KeyType* all_keys;
    GradType* all_grads;

    std::shared_ptr<memory::Allocation> local_keys_mem;
    std::shared_ptr<memory::Allocation> local_grads_mem;
    KeyType* local_keys;
    GradType* local_grads;
  };

  struct SpiltKeysInfo {
    memory::allocation::AllocationPtr d_uniq_size_1;
    memory::allocation::AllocationPtr d_revert_info_1;
    memory::allocation::AllocationPtr d_sort_idxs;
    memory::allocation::AllocationPtr h_uniq_size;
    memory::allocation::AllocationPtr h_shard_left;
    memory::allocation::AllocationPtr h_shard_right;
    memory::allocation::AllocationPtr h_normal_size;
  };

  void init_path();

  void create_storage(int start_index, int end_index, size_t keylen, size_t vallen);
  void destroy_storage(int start_index, int end_index);
  template<typename KeyInfo>
  void walk_to_dest(int start_index, int gpu_num, int* h_left, int* h_right,
                    KeyInfo* src_key, GradType* src_val);
  template <typename KeyInfo>
  void walk_to_dest(int start_index, int gpu_num, int* h_left, int* h_right,
                    KeyInfo* src_key, char* src_val, size_t val_size);
  void walk_to_src(int start_index, int gpu_num, int* h_left, int* h_right,
                   ValType* src_val);
  void walk_to_src(int start_index, int gpu_num, int* h_left, int* h_right,
                   char* src_val, size_t val_size);
  void lxch_barrir();
 protected:
  using Table = HashTable<KeyType, ValType>;
  using PtrTable = HashTable<KeyType, ValType*>;
  std::vector<Table*> tables_;
  std::vector<PtrTable*> ptr_tables_;
  std::atomic<uint64_t> barrir_1_;
  std::atomic<uint64_t> barrir_2_;
  std::mutex mutex_;
  //0 老的， 1 新的, 2 做diff
  uint32_t lxch_run_flag_ {1};
  std::shared_ptr<HeterPsResource> resource_;
  std::vector<std::vector<Path>> path_;
  float load_factor_{0.21};
  int block_size_{256};

 private:
  std::vector<SpiltKeysInfo> split_key_info_;
  std::vector<LocalStorage> storage_;
  CustomGradMerger merger_;
  int topo_aware_{0};
  int feanum_{1800 * 2048};
  int multi_node_{0};
  std::vector<ncclComm_t> nccl_inner_comms_;
  std::vector<ncclComm_t> nccl_inter_comms_;
  int node_size_;
  std::vector<std::shared_ptr<cub::CachingDeviceAllocator>> allocators_;
  int multi_mf_dim_{8};
  int max_mf_dim_ = 8;
};

}  // end namespace framework
}  // end namespace paddle
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_inl.h"
#endif
