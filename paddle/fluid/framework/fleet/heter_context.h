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

//#define likely(x) __builtin_expect(!!(x), 1)
//#define unlikely(x) __builtin_expect(!!(x), 0)

template<typename _Tp>
class FixedMempool {
public:
    FixedMempool(size_t chunk_elemts) : chunk_elemts_(chunk_elemts), cur_chunk_(NULL), cur_index_(0), chunk_list_index_(0) {}
    ~FixedMempool() {
        for (auto iter : chunk_list_) {
            free(iter);
        }
    }
    //获得一个元素
    _Tp* get_one_element() {
        //没有可用元素，分配一个chunk
        if (unlikely(cur_chunk_ == NULL || cur_index_ >= chunk_elemts_)) {
            if (chunk_list_index_ < chunk_list_.size()) {
                cur_chunk_ = chunk_list_[chunk_list_index_];
                chunk_list_index_++;
            } else {
                _Tp* tmp = NULL;
                posix_memalign((void**)&tmp, alignof(_Tp), sizeof(_Tp) * chunk_elemts_);
                chunk_list_.push_back(tmp);
                cur_chunk_ = tmp;
                chunk_list_index_++;
            }
            cur_index_ = 0;
        }
        size_t pre = cur_index_;
        cur_index_++;
        return cur_chunk_ + pre;
    }
    //批量获取元素
    _Tp* get_batch_element(size_t batch_num) {
        if (unlikely(cur_chunk_ == NULL || cur_index_ + batch_num > chunk_elemts_)) {
            if (chunk_list_index_ < chunk_list_.size()) {
                cur_chunk_ = chunk_list_[chunk_list_index_];
                chunk_list_index_++;
            } else {
                _Tp* tmp = NULL;
                posix_memalign((void**)&tmp, alignof(_Tp), sizeof(_Tp) * chunk_elemts_);
                chunk_list_.push_back(tmp);
                cur_chunk_ = tmp;
                chunk_list_index_++;
            }
            cur_index_ = 0;
        }
        size_t pre = cur_index_;
        cur_index_ += batch_num;
        return cur_chunk_ + pre;
    }
    //占用的内存大小
    size_t get_size() {
        return chunk_list_.size() * sizeof(_Tp) * chunk_elemts_;
    }
    void reset() {
        cur_chunk_ = NULL;
        cur_index_ = 0;
        chunk_list_index_ = 0;
    }
private:
    size_t chunk_elemts_; //每个块持有的元素个数
    _Tp* cur_chunk_; //当前块
    size_t cur_index_; //当前块索引
    std::vector<_Tp*> chunk_list_; //块集合
    size_t chunk_list_index_;
};

struct DupUnit {
    uint64_t key;
    DupUnit* next;
};

class KeyDupUnit {
public:
    KeyDupUnit(size_t shard_size) : mempool_(shard_size) {
        local_bucket_ = shard_size;
        buckets_ = mempool_.get_batch_element(local_bucket_);
        memset(buckets_, 0, sizeof(DupUnit) * local_bucket_);
        uniq_keys_num_ = 0;
    } 
    void batch_add_keys(const std::vector<uint64_t>& keys) {
        for (auto key : keys) {
            if (unlikely(key == 0)) {
                continue;
            }
            auto local_id = key % local_bucket_;
            if (buckets_[local_id].key == 0) {
                buckets_[local_id].key = key;
                uniq_keys_num_++;
                continue;
            }
            auto* pre_element = buckets_ + local_id;
            auto* cur_element = buckets_ + local_id;
            while (cur_element != NULL && cur_element->key != key) {
                pre_element = cur_element;
                cur_element = cur_element->next;
            }
            if (cur_element == NULL) {
                cur_element = mempool_.get_one_element();
                cur_element->key = key;
                cur_element->next = NULL;
                pre_element->next = cur_element;
                uniq_keys_num_++;
            }
        }
    }
    size_t get_uniq_key_size() {
        return uniq_keys_num_;
    }
    void trans_to_array(uint64_t* key_start) {
        for (size_t i = 0; i < local_bucket_; i++) {
            if (buckets_[i].key == 0) {
                continue;
            }
            *key_start++ = buckets_[i].key;
            auto cur_element = buckets_[i].next;
            while (cur_element != nullptr) {
                *key_start++ = cur_element->key;
                cur_element = cur_element->next;
            }
        }
    }
    void reset() {
        mempool_.reset();
        uniq_keys_num_ = 0;
        buckets_ = mempool_.get_batch_element(local_bucket_);
        memset(buckets_, 0, sizeof(DupUnit) * local_bucket_);
    }
    void reset(size_t bucket_size) {
        local_bucket_ = bucket_size;
        reset();
    }
    size_t get_bucket_size() {
        return local_bucket_;   
    }
private:
    FixedMempool<DupUnit> mempool_; //内存分配器
    DupUnit* buckets_; //bucket
    size_t local_bucket_; //本地bucket个数
    size_t uniq_keys_num_; //去重后的key数量 
};

class KeyDup {
public:
    KeyDup() {};
    ~KeyDup() {
        clear();
    }
    void init(size_t shard_size, size_t bucket_size) {
        shard_size_ = shard_size;
        shard_ = (KeyDupUnit*) operator new(shard_size_ * sizeof(KeyDupUnit));
        for (size_t i = 0; i < shard_size_; i++) {
            new (shard_ + i)KeyDupUnit(bucket_size);
        }
        max_bucket_count_ = bucket_size;
    }
    void batch_add_keys(size_t shard_id, const std::vector<uint64_t>& keys) {
        shard_[shard_id].batch_add_keys(keys);
    }
    size_t get_uniq_key_size(size_t shard_id) {
        return shard_[shard_id].get_uniq_key_size();
    }
    void trans_to_array(size_t shard_id, uint64_t* key_start) {
        return shard_[shard_id].trans_to_array(key_start);
    }
    void reset() {
        double max_factor = 0;
        size_t bucket_size = 0;
        for (size_t i = 0; i < shard_size_; i++) {
            bucket_size = shard_[i].get_bucket_size();
            auto uniqkeys = shard_[i].get_uniq_key_size();
            double factor = uniqkeys / 1.0 / bucket_size;
            max_factor = std::max(max_factor, factor);
        }
        if (max_factor > 1.2) {
            size_t new_bucket_size = (size_t)(bucket_size * (max_factor + 0.5));
            if (new_bucket_size < 10000) new_bucket_size = 10000;
            new_bucket_size = new_bucket_size / 10 * 10 + 1;
            if (new_bucket_size > max_bucket_count_) {
                clear();
                init(shard_size_, new_bucket_size);
            } else {
                shard_reset(new_bucket_size);
            }
        } else if (max_factor < 0.5) {
            size_t new_bucket_size = (size_t)(bucket_size * 0.5);
            if (new_bucket_size < 10000) new_bucket_size = 10000;
            new_bucket_size = new_bucket_size / 10 * 10 + 1;
            shard_reset(new_bucket_size);
        } else {
            shard_reset(0);
        }
    }
    bool is_init() {
        return shard_size_ != 0;
    }
private:
    void shard_reset(size_t bucket_size) {
        if (bucket_size == 0) {
            for (size_t i = 0; i < shard_size_; i++) {
                shard_[i].reset();
            }
        } else {
            for (size_t i = 0; i < shard_size_; i++) {
                shard_[i].reset(bucket_size);
            }
        }
    }
    void clear() {
        if (shard_ != NULL) {
            delete []shard_;
        }
    }
    KeyDupUnit* shard_ = NULL;
    size_t shard_size_ = 0;
    size_t max_bucket_count_ = 0;
};

class HeterContext {
 public:
  //保存去重后的待查table的key, 第一层对应table-shard, 第二层对应不同维度，第三层就是key集合
  std::vector<std::vector<std::vector<FeatureKey>>>feature_keys_;
  //保存查到的value数据，维度同feature_keys_
#ifdef PADDLE_WITH_PSLIB
  std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>
      value_ptr_;
#endif
#ifdef PADDLE_WITH_PSCORE
  std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>
      value_ptr_;
#endif
  //经过去重后的gpu-table中的key数据, 第一层设备，第二层维度，第三层具体的key
  std::vector<std::vector<std::vector<FeatureKey>>> device_keys_;

  //初始化
  void init(int shard_num, int device_num, int dim_num) {
    feature_keys_.resize(shard_num);
    for (auto& iter : feature_keys_) {
      iter.resize(dim_num);
      for (auto& iter1: iter) {
        iter1.clear();
      }
    }
    value_ptr_.resize(shard_num);
    for (auto& iter : value_ptr_) {
      iter.resize(dim_num);
      for (auto& iter1: iter) {
        iter1.clear();
      }
    }
    device_keys_.resize(device_num);
    for (auto& iter : device_keys_) {
      iter.resize(dim_num);
      for (auto& iter1: iter) {
        iter1.clear();
      }
    }

  }
  //将粗去重的key加入进来,后面再做精细化去重
  void batch_add_keys(int shard_num, int dim_id,
                      const robin_hood::unordered_set<uint64_t>& shard_keys) {
    int idx = feature_keys_[shard_num][dim_id].size();
    feature_keys_[shard_num][dim_id].resize(
        feature_keys_[shard_num][dim_id].size() + shard_keys.size());
    std::copy(shard_keys.begin(), shard_keys.end(),
              feature_keys_[shard_num][dim_id].begin() + idx);
  }
  void unique_keys() {
    std::vector<std::thread> threads;
    auto unique_func = [this](int i, int j) {
      auto& cur_keys = feature_keys_[i][j];
      std::sort(cur_keys.begin(), cur_keys.end());
      std::vector<FeatureKey>::iterator it;
      it = std::unique(cur_keys.begin(), cur_keys.end());
      cur_keys.resize(std::distance(cur_keys.begin(), it));
    };
    for (size_t i = 0; i < feature_keys_.size(); i++) {
      for (size_t j = 0; j < feature_keys_[i].size(); j++) {
        threads.push_back(std::thread(unique_func, i, j));
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
  }
  uint16_t pass_id_;
};


}  // end namespace framework
}  // end namespace paddle
#endif
