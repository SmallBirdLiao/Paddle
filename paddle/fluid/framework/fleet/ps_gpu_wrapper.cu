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
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace framework {

/*lxchtodo
* feature_value_inl.h
*/
template <typename ValueType>
__global__ void kernel_value_to_cvm(float** dest, ValueType* src, FeatureKey** keys, const int slot_num,
                              const int64_t* len, const int* slot_dim, int64_t total_len, int hidden_size, int value_size) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    int cur_dim =hidden_size - 3;
    //动态维度
    if (slot_dim != nullptr) {
      cur_dim = slot_dim[x] - 3;
    }  
    char* p_src = (char*)(src);
    ValueType* value_ptr = (ValueType*)(p_src + uint64_t(i) * uint64_t(value_size));
    if (*(keys[x] + y) == 0) {
      *(dest[x] + y * (cur_dim + 3)) = 0;
      *(dest[x] + y * (cur_dim + 3) + 1) = 0;
      *(dest[x] + y * (cur_dim + 3) + 2) = 0;
      for (int j = 0; j < cur_dim; j++) {
        *(dest[x] + y * (cur_dim + 3) + 3 + j) = 0;
      }
    } else {
      value_ptr->to_cvm(dest[x] + y * (cur_dim + 3), cur_dim);
    }
  }
}

template <typename PushValueType>
__global__ void kernel_grad_to_push(PushValueType* des, float** src, const int slot_num, const int64_t* len,
                                    const int* slot_dim, int64_t total_len, int hidden_size, int value_size,
                                    int batch_size, const int* slot_vector) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[low - 1] : 0);
    char* d_src = (char*)(des);
    PushValueType* value_ptr = (PushValueType*)(d_src + i * value_size);
    int mf_dim = hidden_size - 3; 
    if (slot_dim != nullptr) {
      mf_dim = slot_dim[x];
    }
    int slot_id = slot_vector[x];
    value_ptr->from_grad(src[x] + y * (mf_dim + 3), mf_dim, slot_id, batch_size);
  }
}

class ValueTransforImp : public ValueTransfor {
protected:
  template <typename ValueType>
  void value_to_cvm_impl( float** gpu_cvm,
                          ValueType* gpu_value,
                          FeatureKey** gpu_keys,
                          const int slot_num,
                          const int64_t* key_len,
                          const int* slot_dim,
                          int64_t total_length,
                          int hidden_size,
                          int value_size,
                          cudaStream_t stream) {
    kernel_value_to_cvm<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
        gpu_cvm, gpu_value, gpu_keys, slot_num, key_len, slot_dim, total_length, hidden_size, value_size);
    cudaStreamSynchronize(stream);
  }
  template <typename PushValueType>
  void grad_to_push_impl(PushValueType* push_value,
                            float** grad_value,
                            const int slot_num,
                            const int64_t* grad_len,
                            const int* slot_dim,
                            int64_t total_length,
                            int hidden_size,
                            int value_size,
                            int batch_size,
                            const int* slot_vector,
                            cudaStream_t stream
                            ) {
    kernel_grad_to_push<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
          (PushValueType*)push_value, grad_value, slot_num, grad_len, slot_dim,
                       total_length, hidden_size, value_size, batch_size, slot_vector);
    cudaStreamSynchronize(stream);
  }
};

/*lxchtodo
* gpu_value_inl.h
*/
class T_GpuValue_DownpourCtrAccessor : public ValueTransforImp {
public:
  virtual int get_gpu_value_size(int dim_size) {
    int ret = sizeof(GpuValue);
    return TYPE_ALIGN(8, ret);
  }
  virtual int get_gpu_push_value_size(int dim_size) {
    int ret = sizeof(GpuPushValue);
    return TYPE_ALIGN(8, ret);
  }
  virtual void value_cpu_to_gpu(void* cpu, void* gpu, int dim_size) {
#ifdef PADDLE_WITH_PSLIB
    paddle::ps::DownpourFixedFeatureValue* cpu_value = (paddle::ps::DownpourFixedFeatureValue*)cpu;
    GpuValue* gpu_value = (GpuValue*)gpu;
    const float* ptr_cpu_data = cpu_value->data();
    size_t dim = cpu_value->size();
    gpu_value->delta_score = ptr_cpu_data[1];
    gpu_value->show = ptr_cpu_data[2];
    gpu_value->clk = ptr_cpu_data[3];
    gpu_value->slot = ptr_cpu_data[6];
    gpu_value->lr = ptr_cpu_data[4];
    gpu_value->lr_g2sum = ptr_cpu_data[5];
    gpu_value->cpu_ptr = (uint64_t)(cpu);
    if (dim > 7) {
      gpu_value->mf_size = 9;
      for (int x = 0; x < gpu_value->mf_size; x++) {
        gpu_value->mf[x] = ptr_cpu_data[x + 7];
      }
    } else {
      gpu_value->mf_size = 0;
      for (int x = 0; x < 9; x++) {
        gpu_value->mf[x] = 0;
      }
    }
#endif
#ifdef PADDLE_WITH_PSCORE
    const paddle::distributed::FixedFeatureValue* cpu_value = (const paddle::distributed::FixedFeatureValue*)cpu;
    GpuValue* gpu_value = (GpuValue*)gpu;
    const float* ptr_cpu_data = cpu_value->data();
    size_t dim = cpu_value->size();
    gpu_value->delta_score = ptr_cpu_data[2];
    gpu_value->show = ptr_cpu_data[3];
    gpu_value->clk = ptr_cpu_data[4];
    gpu_value->slot = ptr_cpu_data[0];
    gpu_value->lr = ptr_cpu_data[5];
    gpu_value->lr_g2sum = ptr_cpu_data[6];
    gpu_value->cpu_ptr = (uint64_t)(cpu);
    if (dim > 7) {
      gpu_value->mf_size = 9;
      for (int x = 0; x < gpu_value->mf_size; x++) {
        gpu_value->mf[x] = ptr_cpu_data[x + 7];
      }
    } else {
      gpu_value->mf_size = 0;
      for (int x = 0; x < 9; x++) {
        gpu_value->mf[x] = 0;
      }
    }
#endif
  }
  virtual void value_gpu_to_cpu(void* gpu) {
#ifdef PADDLE_WITH_PSLIB
    GpuValue* gpu_value = (GpuValue*)gpu;
    paddle::ps::DownpourFixedFeatureValue& cpu_fix = *((paddle::ps::DownpourFixedFeatureValue*)(gpu_value->cpu_ptr));
    if (gpu_value->mf_size > 0) {
      cpu_fix.resize(7 + gpu_value->mf_size);
    }
    float* cpu_value = cpu_fix.data();
    cpu_value[1] = gpu_value->delta_score;
    cpu_value[2] = gpu_value->show;
    cpu_value[3] = gpu_value->clk;
    cpu_value[4] = gpu_value->lr;
    cpu_value[5] = gpu_value->lr_g2sum;
    cpu_value[6] = gpu_value->slot;
    if (gpu_value->mf_size > 0) {
       for (int x = 0; x < gpu_value->mf_size; x++) {
         cpu_value[x + 7] = gpu_value->mf[x];
       }
    }
#endif
#ifdef PADDLE_WITH_PSCORE
    GpuValue* gpu_value = (GpuValue*)gpu;
    paddle::distributed::FixedFeatureValue& cpu_value = *((paddle::distributed::FixedFeatureValue*)(gpu_value->cpu_ptr));
    if (gpu_value->mf_size > 0) {
      cpu_value.resize(7 + gpu_value->mf_size);
    }
    cpu_value[2] = gpu_value->delta_score;
    cpu_value[3] = gpu_value->show;
    cpu_value[4] = gpu_value->clk;
    cpu_value[5] = gpu_value->lr;
    cpu_value[6] = gpu_value->lr_g2sum;
    cpu_value[0] = gpu_value->slot;
    if (gpu_value->mf_size > 0) {
       for (int x = 0; x < gpu_value->mf_size; x++) {
         cpu_value[x + 7] = gpu_value->mf[x];
       }
    }
#endif
}
  virtual void value_to_cvm(float** gpu_cvm,
                            const void* gpu_value,
                            FeatureKey** gpu_keys,
                            const int slot_num,
                            const int64_t* key_len,
                            const int* slot_dim,
                            int64_t total_length,
                            int hidden_size,
                            int value_size,
                            cudaStream_t stream
                            ) {
    value_to_cvm_impl(gpu_cvm, (GpuValue*)gpu_value, gpu_keys, slot_num, key_len,
                        slot_dim, total_length, hidden_size, value_size, stream);
  }
  virtual void grad_to_push(void* push_value,
                            float** grad_value,
                            const int slot_num,
                            const int64_t* grad_len,
                            const int* slot_dim,
                            int64_t total_length,
                            int hidden_size,
                            int value_size,
                            int batch_size,
                            const int* slot_vector,
                            cudaStream_t stream
                            ) {
    grad_to_push_impl((GpuPushValue*)push_value, grad_value, slot_num, grad_len, slot_dim, 
                      total_length, hidden_size, value_size, batch_size, slot_vector, stream);
  }
};


/*lxchtodo
* dy_gpu_value_inl.h
*/
class T_DyGpuValue_DownpourCtrDymfAccessor : public ValueTransforImp {
public:
  virtual int get_gpu_value_size(int dim_size) {
    int ret = sizeof(DyGpuValue) + (dim_size + 1) * sizeof(float);
    return TYPE_ALIGN(8, ret);
  }
  virtual int get_gpu_push_value_size(int dim_size) {
    int ret = sizeof(DyGpuPushValue) + (dim_size) * sizeof(float);
    return TYPE_ALIGN(8, ret);
  }
  virtual void value_cpu_to_gpu(void* cpu, void* gpu, int dim_size) {
#ifdef PADDLE_WITH_PSLIB
    paddle::ps::DownpourFixedFeatureValue* cpu_value = (paddle::ps::DownpourFixedFeatureValue*)cpu;
    DyGpuValue* gpu_value = (DyGpuValue*)gpu;
    const float* ptr_cpu_data = cpu_value->data();
    size_t dim = cpu_value->size();
    uint64_t tmp_aa = (uint64_t)(cpu);
    gpu_value->delta_score = ptr_cpu_data[1];
    gpu_value->show = ptr_cpu_data[2];
    gpu_value->clk = ptr_cpu_data[3];
    gpu_value->slot = int(ptr_cpu_data[6]);
    gpu_value->lr = ptr_cpu_data[4];
    gpu_value->lr_g2sum = ptr_cpu_data[5];
    gpu_value->cpu_ptr = (uint64_t)(cpu);
    gpu_value->mf_dim = dim_size;
    if (dim > 8) {
        gpu_value->mf_size = dim_size + 1;
        for (int x = 0; x < gpu_value->mf_dim + 1; x++) {
          gpu_value->mf[x] = ptr_cpu_data[x + 8];
        }
      } else {
        gpu_value->mf_size = 0;
        for (int x = 0; x < gpu_value->mf_dim + 1; x++) {
          gpu_value->mf[x] = 0 ;
        }
      }
#endif
  }
  virtual void value_gpu_to_cpu(void* gpu) {
#ifdef PADDLE_WITH_PSLIB
    DyGpuValue* gpu_value = (DyGpuValue*)gpu;
    paddle::ps::DownpourFixedFeatureValue& cpu_fix = *((paddle::ps::DownpourFixedFeatureValue*)(gpu_value->cpu_ptr));
    if (gpu_value->mf_size > 0) {
      cpu_fix.resize(8 + 1 + gpu_value->mf_dim);
    }
    float* cpu_value = cpu_fix.data();
    cpu_value[1] = gpu_value->delta_score;
    cpu_value[2] = gpu_value->show;
    cpu_value[3] = gpu_value->clk;
    cpu_value[4] = gpu_value->lr;
    cpu_value[5] = gpu_value->lr_g2sum;
    cpu_value[6] = gpu_value->slot;
    if (gpu_value->mf_size > 0) {
       for (int x = 0; x < gpu_value->mf_dim + 1; x++) {
         cpu_value[x + 8] = gpu_value->mf[x];
       }
    }
#endif
  }
  virtual void value_to_cvm(float** gpu_cvm,
                            const void* gpu_value,
                            FeatureKey** gpu_keys,
                            const int slot_num,
                            const int64_t* key_len,
                            const int* slot_dim,
                            int64_t total_length,
                            int hidden_size,
                            int value_size,
                            cudaStream_t stream
                            ) {
    value_to_cvm_impl(gpu_cvm, (DyGpuValue*)gpu_value, gpu_keys, slot_num, key_len,
                        slot_dim, total_length, hidden_size, value_size, stream);
  }
  virtual void grad_to_push(void* push_value,
                            float** grad_value,
                            const int slot_num,
                            const int64_t* grad_len,
                            const int* slot_dim,
                            int64_t total_length,
                            int hidden_size,
                            int value_size,
                            int batch_size,
                            const int* slot_vector,
                            cudaStream_t stream
                            ) {
    grad_to_push_impl((DyGpuPushValue*)push_value, grad_value, slot_num, grad_len, slot_dim,
                       total_length, hidden_size, value_size, batch_size, slot_vector, stream);
  }
};

void GlobalValueTransfor::init(std::string accessor_type, std::string gpu_value_type) {
  if (transobj_ != nullptr) {
    return;
  }
  if (accessor_type == "DownpourCtrDymfAccessor" && gpu_value_type == "DyFeatureValue") {
    transobj_ = (ValueTransfor*)(new T_DyGpuValue_DownpourCtrDymfAccessor());
  } else if (accessor_type == "DownpourCtrAccessor" && gpu_value_type == "FeatureValue") {
    transobj_ = (ValueTransfor*)(new T_GpuValue_DownpourCtrAccessor());
  }
  return;
}

ValueTransfor* GlobalValueTransfor::get_value_transfor() {
  return transobj_;
}


__global__ void CopyKeysKernel(uint64_t** src_keys, uint64_t* dest_total_keys,
                               const int64_t* len, int slot_num,
                               int total_len) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    dest_total_keys[i] = src_keys[x][y];
  }
}

void PSGPUWrapper::CopyKeys(const paddle::platform::Place& place,
                            uint64_t** origin_keys, uint64_t* total_keys,
                            const int64_t* gpu_len, int slot_num,
                            int total_len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  CopyKeysKernel<<<(total_len + 1024 - 1) / 1024, 1024, 0, stream>>>(
      origin_keys, total_keys, gpu_len, slot_num, total_len);
  cudaStreamSynchronize(stream);
}

void PSGPUWrapper::SetSparseSGD(float nonclk_coeff, float clk_coeff,
                                float min_bound, float max_bound,
                                float learning_rate, float initial_g2sum,
                                float initial_range) {
  cudaMemcpyToSymbol(optimizer_config::nonclk_coeff, &nonclk_coeff,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::clk_coeff, &clk_coeff, sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::min_bound, &min_bound, sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::max_bound, &max_bound, sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::learning_rate, &learning_rate,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::initial_g2sum, &initial_g2sum,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::initial_range, &initial_range,
                     sizeof(float));
}

void PSGPUWrapper::SetEmbedxSGD(float mf_create_thresholds,
                                float mf_learning_rate, float mf_initial_g2sum,
                                float mf_initial_range, float mf_min_bound,
                                float mf_max_bound) {
  cudaMemcpyToSymbol(optimizer_config::mf_create_thresholds,
                     &mf_create_thresholds, sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_learning_rate, &mf_learning_rate,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_initial_g2sum, &mf_initial_g2sum,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_initial_range, &mf_initial_range,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_min_bound, &mf_min_bound,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_max_bound, &mf_max_bound,
                     sizeof(float));
}

}  // end namespace framework
}  // end namespace paddle
#endif
