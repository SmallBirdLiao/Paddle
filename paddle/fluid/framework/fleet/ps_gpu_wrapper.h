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

#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef PADDLE_WITH_GLOO
#include <gloo/broadcast.h>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif
#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/heter_context.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_ps_base.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/mem_pool.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#endif
#ifdef PADDLE_WITH_PSLIB
#include "afs_api.h"
#endif
#ifdef PADDLE_WITH_PSLIB
#include "downpour_accessor.h"  // NOLINT
#endif

namespace paddle {
namespace framework {

#define TYPEALIGN(ALIGNVAL, LEN) \
  (((uint64_t)(LEN) + ((ALIGNVAL)-1)) & ~((uint64_t)((ALIGNVAL)-1)))

#ifdef PADDLE_WITH_PSLIB
class AfsWrapper {
 public:
  AfsWrapper() {}
  virtual ~AfsWrapper() {}
  void init(const std::string& fs_name, const std::string& fs_user,
            const std::string& pass_wd, const std::string& conf);
  int remove(const std::string& path);
  int mkdir(const std::string& path);
  std::vector<std::string> list(const std::string& path);

  int exist(const std::string& path);
  int upload(const std::string& local_file, const std::string& afs_file);

  int download(const std::string& local_file, const std::string& afs_file);

  int touchz(const std::string& path);
  std::string cat(const std::string& path);
  int mv(const std::string& old_path, const std::string& dest_path);

 private:
  paddle::ps::AfsApiWrapper afs_handler_;
};
#endif

class PSGPUWrapper {
 public:
  virtual ~PSGPUWrapper() { delete HeterPs_; }

  PSGPUWrapper() {
    HeterPs_ = NULL;
    sleep_seconds_before_fail_exit_ = 300;
    ps_accessor_type_ = "DownpourCtrAccessor";
    gpu_value_type_ = "FeatureValue";
  }

  void PullSparse(const paddle::platform::Place& place, const int table_id,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const std::vector<int>& slot_dim, const int hidden_size);
  void PullSparse(const paddle::platform::Place& place, const int table_id,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place, const int table_id,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size, const int batch_size);
  void CopyKeys(const paddle::platform::Place& place, uint64_t** origin_keys,
                uint64_t* total_keys, const int64_t* gpu_len, int slot_num,
                int total_len);
  void CopyKeys(const paddle::platform::Place& place, uint64_t** origin_keys,
                uint64_t* total_keys, const int64_t* gpu_len, int slot_num,
                int total_len, int* gpu_dim);

  void BuildGPUTask(std::shared_ptr<HeterContext> gpu_task);
  void PreBuildTask(std::shared_ptr<HeterContext> gpu_task);
  void BuildPull(std::shared_ptr<HeterContext> gpu_task);
  void LoadIntoMemory(bool is_shuffle);
  void BeginPass();
  void EndPass();
  void start_build_thread();
  void pre_build_thread();
  void build_pull_thread();
  void build_task();

  void Finalize() {
    VLOG(3) << "PSGPUWrapper Begin Finalize.";
    if (s_instance_ == nullptr) {
      return;
    }
    data_ready_channel_->Close();
    buildcpu_ready_channel_->Close();
    buildpull_ready_channel_->Close();
    gpu_free_channel_->Close();
    running_ = false;
    VLOG(3) << "begin stop pre_build_threads_";
    pre_build_threads_.join();
    VLOG(3) << "begin stop buildpull_threads_";
    buildpull_threads_.join();
    s_instance_ = nullptr;
    VLOG(3) << "PSGPUWrapper Finalize Finished.";
  }

  void InitializeGPU(const std::vector<int>& dev_ids) {
    if (s_instance_ != NULL && is_initialized_ == false) {
      VLOG(3) << "PSGPUWrapper Begin InitializeGPU";
      is_initialized_ = true;
      resource_ = std::make_shared<HeterPsResource>(dev_ids);
      resource_->enable_p2p();
      keys_tensor_.resize(resource_->total_gpu());
#ifdef PADDLE_WITH_GLOO
      auto gloo = paddle::framework::GlooWrapper::GetInstance();
      if (gloo->Size() > 1) {
        multi_node_ = 1;
      }
#else
      PADDLE_THROW(
          platform::errors::Unavailable("heter ps need compile with GLOO"));
#endif
      if (multi_node_) {
        int dev_size = dev_ids.size();
        // init inner comm
        inner_comms_.resize(dev_size);
        inter_ncclids_.resize(dev_size);
        platform::dynload::ncclCommInitAll(&(inner_comms_[0]), dev_size,
                                           &dev_ids[0]);
// init inter comm
#ifdef PADDLE_WITH_GLOO
        inter_comms_.resize(dev_size);
        if (gloo->Rank() == 0) {
          for (int i = 0; i < dev_size; ++i) {
            platform::dynload::ncclGetUniqueId(&inter_ncclids_[i]);
          }
        }

        PADDLE_ENFORCE_EQ(
            gloo->IsInitialized(), true,
            platform::errors::PreconditionNotMet(
                "You must initialize the gloo environment first to use it."));
        gloo::BroadcastOptions opts(gloo->GetContext());
        opts.setOutput(&inter_ncclids_[0], dev_size);
        opts.setRoot(0);
        gloo::broadcast(opts);

        for (int i = 0; i < dev_size; ++i) {
          platform::dynload::ncclCommInitRank(&inter_comms_[i], gloo->Size(),
                                              inter_ncclids_[i], gloo->Rank());
        }
        node_size_ = gloo->Size();
#else
        PADDLE_THROW(
            platform::errors::Unavailable("heter ps need compile with GLOO"));
#endif
      }
      heter_devices_ = dev_ids;
      data_ready_channel_->Open();
      data_ready_channel_->SetCapacity(3);
      buildcpu_ready_channel_->Open();
      buildcpu_ready_channel_->SetCapacity(3);
      buildpull_ready_channel_->Open();
      buildpull_ready_channel_->SetCapacity(1);
      gpu_free_channel_->Open();
      gpu_free_channel_->SetCapacity(1);

      current_task_ = nullptr;
      gpu_free_channel_->Put(current_task_);

      table_id_ = 0;

      // start build cpu&gpu ps thread
      start_build_thread();
    }
  }

  void SetSparseSGD(float nonclk_coeff, float clk_coeff, float min_bound,
                    float max_bound, float learning_rate, float initial_g2sum,
                    float initial_range);
  void SetEmbedxSGD(float mf_create_thresholds, float mf_learning_rate,
                    float mf_initial_g2sum, float mf_initial_range,
                    float mf_min_bound, float mf_max_bound);
  void InitializeGPUServer(std::unordered_map<std::string, float> config) {
    float nonclk_coeff = (config.find("nonclk_coeff") == config.end())
                             ? 1.0
                             : config["nonclk_coeff"];
    float clk_coeff =
        (config.find("clk_coeff") == config.end()) ? 1.0 : config["clk_coeff"];
    float min_bound = (config.find("min_bound") == config.end())
                          ? -10000.0
                          : config["min_bound"];
    float max_bound = (config.find("max_bound") == config.end())
                          ? 10000.0
                          : config["max_bound"];
    float learning_rate = (config.find("learning_rate") == config.end())
                              ? 1.0
                              : config["learning_rate"];
    float initial_g2sum = (config.find("initial_g2sum") == config.end())
                              ? 1.0
                              : config["initial_g2sum"];
    float initial_range = (config.find("initial_range") == config.end())
                              ? 1.0
                              : config["initial_range"];

    // mf config settings
    float mf_create_thresholds =
        (config.find("mf_create_thresholds") == config.end())
            ? static_cast<float>(1.0)
            : config["mf_create_thresholds"];
    float mf_learning_rate = (config.find("mf_learning_rate") == config.end())
                                 ? 1.0
                                 : config["mf_learning_rate"];
    float mf_initial_g2sum = (config.find("mf_initial_g2sum") == config.end())
                                 ? 1.0
                                 : config["mf_initial_g2sum"];
    float mf_initial_range = (config.find("mf_initial_range") == config.end())
                                 ? 1.0
                                 : config["mf_initial_range"];
    float mf_min_bound = (config.find("mf_min_bound") == config.end())
                             ? 1.0
                             : config["mf_min_bound"];
    float mf_max_bound = (config.find("mf_max_bound") == config.end())
                             ? 1.0
                             : config["mf_max_bound"];
    for (size_t i = 0; i < heter_devices_.size(); i++) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(heter_devices_[i]));
      this->SetSparseSGD(nonclk_coeff, clk_coeff, min_bound, max_bound,
                         learning_rate, initial_g2sum, initial_range);
      this->SetEmbedxSGD(mf_create_thresholds, mf_learning_rate,
                         mf_initial_g2sum, mf_initial_range, mf_min_bound,
                         mf_max_bound);
    }
  }
  void SetDate(int year, int month, int day) {
    year_ = year;
    month_ = month;
    day_ = day;
  }

  void SetDataset(Dataset* dataset) { 
    dataset_ = dataset; 
  }

  // PSGPUWrapper singleton
  static std::shared_ptr<PSGPUWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::PSGPUWrapper());
    }
    return s_instance_;
  }
  void SetSlotVector(const std::vector<int>& slot_vector) {
    slot_vector_ = slot_vector;
  }

  void SetSlotOffsetVector(const std::vector<int>& slot_offset_vector) {
    slot_offset_vector_ = slot_offset_vector;
  }

  void SetSlotDimVector(const std::vector<int>& slot_mf_dim_vector) {
    slot_mf_dim_vector_ = slot_mf_dim_vector;
    assert(slot_mf_dim_vector_.size() == slot_vector_.size());
  }

  void SetSlotDimFixed(const int dim) {
    assert(slot_vector_.size() != 0);
    for (size_t ii = 0; ii < slot_vector_.size(); ii++) {
      slot_mf_dim_vector_.push_back(dim);
    }
  }

  void SetTableShardNum(const int shard_num) {
    thread_keys_shard_num_ = shard_num;
  }

  void SetAccessorAndValueType(std::string accessor_type, std::string value_type) {
    ps_accessor_type_ = accessor_type;
    gpu_value_type_ = value_type;
  }

  void InitSlotInfo() {
    if (slot_info_initialized_) {
      return;
    }
    SlotRecordDataset* dataset = dynamic_cast<SlotRecordDataset*>(dataset_);
    auto slots_vec = dataset->GetSlots();
    slot_offset_vector_.clear();
    for (auto& slot : slot_vector_) {
      for (size_t i = 0; i < slots_vec.size(); ++i) {
        if (std::to_string(slot) == slots_vec[i]) {
          slot_offset_vector_.push_back(i);
          break;
        }
      }
    }

    std::unordered_set<int> dims_set;
    for (auto& it : slot_mf_dim_vector_) {
      dims_set.insert(it);
    }
    size_t num_of_dim = dims_set.size();
    index_dim_vec_.resize(num_of_dim);
    index_dim_vec_.assign(dims_set.begin(), dims_set.end());
    std::sort(index_dim_vec_.begin(), index_dim_vec_.end());
    std::unordered_map<int, int> dim_index_map;
    for (size_t i = 0; i < num_of_dim; i++) {
      dim_index_map[index_dim_vec_[i]] = i;
    }
    hbm_pools_.resize(resource_->total_gpu() * num_of_dim);
    mem_pools_.resize(resource_->total_gpu() * num_of_dim);
    max_mf_dim_ = index_dim_vec_.back();
    multi_mf_dim_ = (dim_index_map.size() >= 1) ? dim_index_map.size() : 0;
    resource_->set_multi_mf(multi_mf_dim_, max_mf_dim_);
    slot_index_vec_.resize(slot_mf_dim_vector_.size());
    for (size_t i = 0; i < slot_index_vec_.size(); i++) {
      slot_index_vec_[i] = dim_index_map[slot_mf_dim_vector_[i]];
    }
    //初始化一些线程池
    CHECK(thread_keys_shard_num_ != 0);
    hbm_thread_pool_.resize(thread_keys_shard_num_);
    for (size_t i = 0; i < hbm_thread_pool_.size(); i++) {
      hbm_thread_pool_[i].reset(new ::ThreadPool(1));
    }
    pull_thread_pool_.resize(thread_keys_shard_num_);
    for (size_t i = 0; i < pull_thread_pool_.size(); i++) {
      pull_thread_pool_[i].reset(new ::ThreadPool(1));
    }

    uniq_thread_pool_.resize(32);
    for (size_t i = 0; i < 32; i++) {
      uniq_thread_pool_[i].reset(new ::ThreadPool(1));
    }

    GlobalValueTransfor::get_instance().init(ps_accessor_type_, gpu_value_type_);
    slot_info_initialized_ = true;
  }

  void ShowOneTable(int index) { HeterPs_->show_one_table(index); }

  int UseAfsApi() { return use_afs_api_; }

#ifdef PADDLE_WITH_PSLIB
  std::shared_ptr<paddle::ps::AfsReader> OpenReader(
      const std::string& filename) {
    return afs_handler_.open_reader(filename);
  }

  void InitAfsApi(const std::string& fs_name, const std::string& fs_user,
                  const std::string& pass_wd, const std::string& conf);
#endif

 private:
  static std::shared_ptr<PSGPUWrapper> s_instance_;
  //主要用于load数据的时候用的
  Dataset* dataset_;
  //当load数据完成后，会将其筛入到如下队列，后续异步pull会用到这个队列的数据
  //因为load 和 异步build是两个线程，所以才需要下面的队列来解耦这个dataset对象
  std::queue<Dataset*> dataset_pipe_;
  std::mutex dataset_mutex_;
#ifdef PADDLE_WITH_PSLIB
  paddle::ps::AfsApiWrapper afs_handler_;
#endif
  HeterPsBase* HeterPs_;
  std::vector<LoDTensor> keys_tensor_;  // Cache for pull_sparse
  std::shared_ptr<HeterPsResource> resource_;
  int32_t sleep_seconds_before_fail_exit_;
  std::vector<int> slot_vector_;
  std::vector<int> slot_offset_vector_;
  std::vector<int> slot_mf_dim_vector_;
  std::vector<int> slot_index_vec_;
  std::vector<int> index_dim_vec_;
  int multi_mf_dim_{0};
  int max_mf_dim_{0};
  int multi_node_{0};
  int node_size_;
  uint64_t table_id_;
  std::vector<ncclComm_t> inner_comms_;
  std::vector<ncclComm_t> inter_comms_;
  std::vector<ncclUniqueId> inter_ncclids_;
  std::vector<int> heter_devices_;
  HeterObjectPool<HeterContext> gpu_task_pool_;
  std::vector<std::vector<std::vector<robin_hood::unordered_set<uint64_t>>>> thread_keys_;
  int thread_keys_thread_num_ = 37 * 4;
  int thread_keys_shard_num_ = 0;
  uint64_t max_fea_num_per_pass_ = 5000000000;
  int year_;
  int month_;
  int day_;
  bool slot_info_initialized_ = false;
  int use_afs_api_ = 0;
  std::string ps_accessor_type_;
  std::string gpu_value_type_;
  std::vector<MemoryPool*> mem_pools_;
  std::vector<HBMMemoryPool*> hbm_pools_;  // in multi mfdim, one table need hbm
                                           // pools of totol dims number

  std::shared_ptr<
      paddle::framework::ChannelObject<std::shared_ptr<HeterContext>>>
      data_ready_channel_ =
          paddle::framework::MakeChannel<std::shared_ptr<HeterContext>>();
  std::shared_ptr<
      paddle::framework::ChannelObject<std::shared_ptr<HeterContext>>>
      buildcpu_ready_channel_ =
          paddle::framework::MakeChannel<std::shared_ptr<HeterContext>>();
  std::shared_ptr<
      paddle::framework::ChannelObject<std::shared_ptr<HeterContext>>>
      gpu_free_channel_ =
          paddle::framework::MakeChannel<std::shared_ptr<HeterContext>>();
  std::shared_ptr<
      paddle::framework::ChannelObject<std::shared_ptr<HeterContext>>>
      buildpull_ready_channel_ =
          paddle::framework::MakeChannel<std::shared_ptr<HeterContext>>();
  std::shared_ptr<HeterContext> current_task_ = nullptr;
  std::thread pre_build_threads_;
  std::thread buildpull_threads_;
  bool running_ = false;
  std::vector<std::shared_ptr<ThreadPool>> hbm_thread_pool_;
  std::vector<std::shared_ptr<ThreadPool>> pull_thread_pool_;

  std::vector<std::shared_ptr<ThreadPool>> uniq_thread_pool_;

 protected:
  static bool is_initialized_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
