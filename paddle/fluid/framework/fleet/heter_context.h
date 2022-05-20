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

#include <ThreadPool.h>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>

#ifdef PADDLE_WITH_PSLIB
#include "common_value.h"  // NOLINT
#endif

#ifdef PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#endif

#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

//pined内存的vector, 方便进行到显存的copy
//只能用于基础类型。
//(不会调用对象内部的构造函数和析构函数，使用的时候请注意)
template <class _T>
class PinnedVector {
public:
  using ValueType = _T;
  PinnedVector() {
    ptr_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }
  PinnedVector(const PinnedVector<_T>& value) = delete;
  PinnedVector& operator=(const PinnedVector<_T>& value) = delete;
  PinnedVector(PinnedVector<_T>&& value) {
    size_ = value.size_;
    capacity_ = value.capacity_;
    std::swap(ptr_, value.ptr_);
    value.data_ = nullptr;
    data_ = reinterpret_cast<ValueType*>(ptr_->ptr());
  }
  PinnedVector& operator=(PinnedVector<_T>&& value) = delete;
  void clear() {
    ptr_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }
  void resize(size_t size) {
    if (size > capacity_) {
      auto ptr = memory::Alloc(phi::GPUPinnedPlace(), size * sizeof(ValueType));
      if (size_ != 0) {
        memcpy(ptr->ptr(), ptr_->ptr(), sizeof(ValueType) * size_);
      }
    
      capacity_ = size;
      std::swap(ptr, ptr_);
      data_ = reinterpret_cast<ValueType*>(ptr_->ptr());
    }
    if (size > size_) {
      /*
      for (size_t i = size_; i < size; i++) {
        new (data_ + i) ValueType();
      }
      */
      size_ = size;
    } else {
      /*
      for (size_t i = size; i < size_; i++) {
        (data_ + i)->~ValueType();
      }
      */
      size_ = size;
    }
  }
  ValueType& operator[](size_t pos) {
    return data_[pos];
  }
  size_t size() {
    return size_;
  }
  size_t  capacity() {
    return capacity_;
  }
  ValueType* data() {
    return data_;
  }
private:
  size_t size_ = 0;
  size_t capacity_ = 0;
  memory::allocation::AllocationPtr ptr_;
  ValueType* data_;
};

class HeterContext {
 public:
  virtual ~HeterContext() {
    if (!multi_mf_dim_) {
      for (size_t i = 0; i < mutex_.size(); ++i) {
        delete mutex_[i];
      }
      mutex_.clear();
    } else {
      for (size_t i = 0; i < dim_mutex_.size(); ++i) {
        for (size_t j = 0; j < dim_mutex_[i].size(); j++) {
          delete dim_mutex_[i][j];
        }
        dim_mutex_[i].clear();
      }
    }
  }
  Scope* scope_{nullptr};
  std::vector<std::vector<FeatureKey>> feature_keys_;
  std::vector<std::vector<std::vector<FeatureKey>>> feature_dim_keys_;
  std::vector<std::vector<std::vector<FeatureKey>>> device_task_keys_;

#ifdef PADDLE_WITH_PSLIB
  std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>> value_ptr_;
  std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>
      device_task_ptr_;
  std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>
      value_dim_ptr_;
  std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>
      device_dim_ptr_;
#endif
#ifdef PADDLE_WITH_PSCORE
  std::vector<std::vector<paddle::distributed::FixedFeatureValue*>> value_ptr_;
  std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>
      value_dim_ptr_;
  std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>
      device_task_ptr_;
  std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>
      device_dim_ptr_;
#endif
  std::vector<PinnedVector<FeatureValue>> device_values_;
  std::vector<PinnedVector<FeatureKey>> device_keys_;
  std::vector<std::vector<std::vector<FeatureKey>>> device_dim_keys_;
  std::vector<std::vector<std::vector<FeatureValue>>> device_dim_values_;
  std::vector<std::mutex*> mutex_;
  std::vector<std::vector<std::mutex*>> dim_mutex_;
  int multi_mf_dim_ = 0;

  uint32_t shard_num_ = 37;
  uint64_t size() {
    uint64_t total_size = 0;
    for (auto& keys : feature_keys_) {
      total_size += keys.size();
    }
    return total_size;
  }
  void SetShardNum(uint32_t shard_num) { shard_num_ = shard_num; }
  uint32_t ShardNum() { return shard_num_; }
  void init(int shard_num, int device_num) {
    shard_num_ = shard_num;
    feature_keys_.resize(shard_num_);
    value_ptr_.resize(shard_num_);
    device_task_ptr_.resize(shard_num_);
    device_task_keys_.resize(shard_num_);
    for (size_t i = 0; i < device_task_ptr_.size(); i++) {
      device_task_ptr_[i].resize(device_num);
      device_task_keys_[i].resize(device_num);
    }

    device_values_.resize(device_num);
    device_keys_.resize(device_num);
    mutex_.resize(device_num);
    for (size_t i = 0; i < mutex_.size(); ++i) {
      mutex_[i] = new std::mutex();
    }
  }

  void init(int shard_num, int device_num, int dim_num) {
    shard_num_ = shard_num;
    feature_keys_.resize(shard_num_);
    feature_dim_keys_.resize(shard_num_);
    value_ptr_.resize(shard_num_);
    value_dim_ptr_.resize(shard_num_);
    device_task_ptr_.resize(shard_num_);
    device_task_keys_.resize(shard_num_);
    for (size_t i = 0; i < device_task_ptr_.size(); i++) {
      device_task_ptr_[i].resize(device_num);
      device_task_keys_[i].resize(device_num);
    }
    for (size_t i = 0; i < feature_dim_keys_.size(); i++) {
      feature_dim_keys_[i].resize(dim_num);
      value_dim_ptr_[i].resize(dim_num);
      if (i == 0) {
        for (int j = 0; j < dim_num; j++) {
          feature_dim_keys_[i][j].push_back(0);
        }
      }
    }
    device_values_.resize(device_num);
    device_dim_values_.resize(device_num);
    device_keys_.resize(device_num);

    device_dim_keys_.resize(device_num);
    device_dim_ptr_.resize(device_num);
    mutex_.resize(device_num);
    dim_mutex_.resize(device_num);
    for (size_t i = 0; i < mutex_.size(); ++i) {
      mutex_[i] = new std::mutex();
    }
    for (size_t i = 0; i < dim_mutex_.size(); ++i) {
      dim_mutex_[i].resize(dim_num);
      for (int j = 0; j < dim_num; j++) {
        dim_mutex_[i][j] = new std::mutex();
      }
    }
    multi_mf_dim_ = dim_num;
  }

  void Reset() {
    if (!multi_mf_dim_) {
      for (size_t i = 0; i < feature_keys_.size(); ++i) {
        feature_keys_[i].clear();
      }
      for (size_t i = 0; i < value_ptr_.size(); ++i) {
        value_ptr_[i].clear();
      }
      for (size_t i = 0; i < device_values_.size(); ++i) {
        device_values_[i].clear();
      }
      for (size_t i = 0; i < device_keys_.size(); ++i) {
        device_keys_[i].clear();
      }
      for (size_t i = 0; i < device_task_ptr_.size(); ++i) {
        for (size_t j = 0; j < device_task_ptr_[i].size(); ++j) {
          device_task_ptr_[i][j].clear();
          device_task_keys_[i][j].clear();
        }
      }
    } else {
      VLOG(3) << "Reset gpu task with dynamic mf dimention";
      for (size_t i = 0; i < feature_dim_keys_.size(); i++) {
        for (size_t j = 0; j < feature_dim_keys_[i].size(); j++) {
          feature_dim_keys_[i][j].clear();
        }
      }
      for (size_t i = 0; i < value_dim_ptr_.size(); i++) {
        for (size_t j = 0; j < value_dim_ptr_[i].size(); j++) {
          value_dim_ptr_[i][j].clear();
        }
      }

      for (size_t i = 0; i < device_dim_keys_.size(); i++) {
        for (size_t j = 0; j < device_dim_keys_[i].size(); j++) {
          device_dim_keys_[i][j].clear();
        }
      }
      for (size_t i = 0; i < device_dim_ptr_.size(); i++) {
        for (size_t j = 0; j < device_dim_ptr_[i].size(); j++) {
          device_dim_ptr_[i][j].clear();
        }
      }
    }
  }
  void batch_add_keys(
      const std::vector<std::unordered_set<uint64_t>>& thread_keys) {
    assert(thread_keys.size() == feature_keys_.size());

    for (uint32_t i = 0; i < shard_num_; i++) {
      int idx = 0;
      idx = feature_keys_[i].size();
      feature_keys_[i].resize(feature_keys_[i].size() + thread_keys[i].size());
      std::copy(thread_keys[i].begin(), thread_keys[i].end(),
                feature_keys_[i].begin() + idx);
    }
  }

  void batch_add_keys(int shard_num,
                      const robin_hood::unordered_set<uint64_t>& shard_keys) {
    int idx = feature_keys_[shard_num].size();
    feature_keys_[shard_num].resize(feature_keys_[shard_num].size() +
                                    shard_keys.size());
    std::copy(shard_keys.begin(), shard_keys.end(),
              feature_keys_[shard_num].begin() + idx);
  }

  void batch_add_keys(int shard_num, int dim_id,
                      const robin_hood::unordered_set<uint64_t>& shard_keys) {
    int idx = feature_dim_keys_[shard_num][dim_id].size();
    feature_dim_keys_[shard_num][dim_id].resize(
        feature_dim_keys_[shard_num][dim_id].size() + shard_keys.size());
    std::copy(shard_keys.begin(), shard_keys.end(),
              feature_dim_keys_[shard_num][dim_id].begin() + idx);
  }

  void UniqueKeys() {
    std::vector<std::thread> threads;
    auto unique_func = [this](int i) {
      auto& cur_keys = feature_keys_[i];
      std::sort(cur_keys.begin(), cur_keys.end());
      std::vector<FeatureKey>::iterator it;
      it = std::unique(cur_keys.begin(), cur_keys.end());
      cur_keys.resize(std::distance(cur_keys.begin(), it));
    };
    auto unique_dynamic_mf_func = [this](int i, int j) {
      auto& cur_keys = feature_dim_keys_[i][j];
      std::sort(cur_keys.begin(), cur_keys.end());
      std::vector<FeatureKey>::iterator it;
      it = std::unique(cur_keys.begin(), cur_keys.end());
      cur_keys.resize(std::distance(cur_keys.begin(), it));
    };
    if (!multi_mf_dim_) {
      for (uint32_t i = 0; i < shard_num_; i++) {
        threads.push_back(std::thread(unique_func, i));
      }
    } else {
      for (uint32_t i = 0; i < shard_num_; i++) {
        for (int j = 0; j < multi_mf_dim_; j++) {
          threads.push_back(std::thread(unique_dynamic_mf_func, i, j));
        }
      }
      VLOG(3) << "heter_context unique keys with dynamic mf dimention";
    }
    for (std::thread& t : threads) {
      t.join();
    }
  }
};

}  // end namespace framework
}  // end namespace paddle
#endif
