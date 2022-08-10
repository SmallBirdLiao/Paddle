/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/seqpool_scatter_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void seqpool_scatter_kernel( const T* input, size_t input_1_size,
                                        const T* idx,  
                                        T* output, size_t dim_0_size, size_t dim_1_size, size_t dim_2_size,
                                        const T pad_value) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t all_kernel = dim_0_size * dim_1_size * dim_2_size;
    if (i < all_kernel) {
        size_t tmp = dim_1_size * dim_2_size;
        size_t batch_index = i / tmp;
        size_t lod_index = (i % tmp) / dim_2_size;
        size_t data_index = (i % tmp) % dim_2_size;
        size_t idx_idx = batch_index * (dim_1_size + 1) + lod_index;
        int start = int(idx[idx_idx] + 0.5);
        int end = int(idx[idx_idx + 1] + 0.5);
        size_t out_index = batch_index * tmp + lod_index * dim_2_size + data_index;
        if (start == end) {
            output[out_index] = pad_value;
        } else {
            output[out_index] = 0;
            for (int i = start; i < end; i++) {
                size_t tmp_index = batch_index * (input_1_size * dim_2_size) + i * dim_2_size + data_index;
                output[out_index] += input[tmp_index];
            }
        }
    }
}

template <typename T>
__global__ void seqpool_scatter_grad_kernel(T* in_g, size_t input_1_size,
                                            const T* idx,
                                            const T* out_g, size_t dim_0_size, size_t dim_1_size, size_t dim_2_size
                                            ) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t all_kernel = dim_0_size * dim_1_size * dim_2_size;
    if (i < all_kernel) {
        size_t tmp = dim_1_size * dim_2_size;
        size_t batch_index = i / tmp;
        size_t lod_index = (i % tmp) / dim_2_size;
        size_t data_index = (i % tmp) % dim_2_size;
        size_t idx_idx = batch_index * (dim_1_size + 1) + lod_index;
        int start = int(idx[idx_idx] + 0.5);
        int end = int(idx[idx_idx + 1] + 0.5);
        size_t out_index = batch_index * tmp + lod_index * dim_2_size + data_index;
        if (start != end) {
            for (int i = start; i < end; i++) {
                size_t tmp_index = batch_index * (input_1_size * dim_2_size) + i * dim_2_size + data_index;
                in_g[tmp_index] = out_g[out_index];
            }
        }
    }
}

template <typename T>
__global__ void seqpool_scatter_grad_reset_kernel(T* in_g, size_t numel_size) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel_size) {
        in_g[i] = 0.0;
    }
}



template <typename T>
class SeqpoolScatterCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in_x = context.Input<LoDTensor>("X");
    auto* in_idx = context.Input<LoDTensor>("Idx");
    auto* out = context.Output<LoDTensor>("Out");
    T pad_value = static_cast<T>(context.Attr<float>("pad_value"));
    auto lod = in_idx->lod();
    PADDLE_ENFORCE_EQ(lod.size(), 1, platform::errors::InvalidArgument(
                                        "Input(Idx) Tensor of SeqpoolScatterOp "
                                        "LodLevel must be 1."));
    PADDLE_ENFORCE_GE(lod[0].size(), 2, platform::errors::InvalidArgument(
                                        "Input(Idx) Tensor of SeqpoolScatterOp "
                                        "lod[0].size() must be large than 2"));
    int dim0 = lod[0][1] - lod[0][0];
    for (size_t i = 1; i < lod[0].size(); i++) {
        int dim0_tmp = lod[0][i] - lod[0][i-1];
        PADDLE_ENFORCE_EQ(dim0, dim0_tmp, platform::errors::InvalidArgument(
                                        "Input(Idx) Tensor of SeqpoolScatterOp "
                                        "each item must have same dim"));
    }
    dim0 -= 1;
    auto dims = in_x->dims();
    dims[1] = dim0;
    out->Resize({dims});
    out->mutable_data<T>(context.GetPlace());
    size_t kernel_all = dims[0] * dims[1] * dims[2];
    size_t thread_kernel = 256;
    size_t block_kernel = kernel_all / thread_kernel + 1;
    seqpool_scatter_kernel<<<dim3(block_kernel), dim3(thread_kernel), 
                        0, context.device_context<paddle::platform::CUDADeviceContext>().stream()>>>(
            in_x->data<T>(),
            in_x->dims()[1],
            in_idx->data<T>(),
            out->data<T>(),
            dims[0], dims[1], dims[2],
            pad_value);
    if (0) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        int tmp_x_size = in_x->numel();
        int tmp_idx_size = in_idx->numel();
        int tmp_out_size = out->numel();
        auto h_x = memory::Alloc(phi::GPUPinnedPlace(), tmp_x_size * sizeof(T));
        T* h_x_ptr = (T*)h_x->ptr();
        auto h_idx = memory::Alloc(phi::GPUPinnedPlace(), tmp_idx_size* sizeof(T));
        T* h_idx_ptr = (T*)h_idx->ptr();
        auto g_out = memory::Alloc(phi::GPUPinnedPlace(), tmp_out_size * sizeof(T));
        T* g_out_ptr = (T*)g_out->ptr();
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyAsync(h_x_ptr, in_x->data<T>(), tmp_x_size * sizeof(T), cudaMemcpyDeviceToHost,
                        context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyAsync(h_idx_ptr, in_idx->data<T>(), tmp_idx_size * sizeof(T), cudaMemcpyDeviceToHost,
                        context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyAsync(g_out_ptr, out->data<T>(), tmp_out_size * sizeof(T), cudaMemcpyDeviceToHost,
                        context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        auto lod_idx = in_idx->lod();
        auto dim_x = in_x->dims();
        auto dim_out = out->dims();
        for (size_t i = 0; i < lod_idx[0].size() - 1; i++) {
            int start = lod_idx[0][i];
            int end = lod_idx[0][i+1];
            int batch_idx = i;
            for (int j = start, k = 0; j < end - 1; j++, k++) {
                int idx_start = int(h_idx_ptr[j] + 0.5);
                int idx_end = int(h_idx_ptr[j+1] + 0.5);
                for (int ii = 0; ii < dim_x[2]; ii++) {
                    float tmp = 0;
                    for (int jj = idx_start; jj < idx_end; jj++) {
                        tmp += *(h_x_ptr + batch_idx * dim_x[1] * dim_x[2] + jj * dim_x[2] + ii);
                    }
                    float tmp_1 = *(g_out_ptr + batch_idx * dim_out[1] * dim_out[2] + k * dim_out[2] + ii);
                    if (fabs(tmp_1 - tmp) >= 0.000001) {
                        abort();
                    }
                }
            }
        }
    }
  }
};


template <typename T>
class SeqpoolScatterGradCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* in_idx = context.Input<LoDTensor>("Idx");
    auto* in_x = context.Input<LoDTensor>("X");
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));

    in_g->Resize({in_x->dims()});
    in_g->mutable_data<T>(context.GetPlace());

    auto dims = out_g->dims();

    size_t kernel_all = dims[0] * dims[1] * dims[2];
    size_t thread_kernel = 256;
    size_t block_kernel = kernel_all / thread_kernel + 1;
    seqpool_scatter_grad_reset_kernel<<<dim3(in_g->numel()/thread_kernel + 1), dim3(thread_kernel), 
                        0, context.device_context<paddle::platform::CUDADeviceContext>().stream()>>>(
            in_g->data<T>(),in_g->numel());

    seqpool_scatter_grad_kernel<<<dim3(block_kernel), dim3(thread_kernel), 
                        0, context.device_context<paddle::platform::CUDADeviceContext>().stream()>>>(
            in_g->data<T>(),
            in_g->dims()[1],
            in_idx->data<T>(),
            out_g->data<T>(),
            dims[0], dims[1], dims[2]);

    if (0) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        auto h_in = memory::Alloc(phi::GPUPinnedPlace(), in_g->numel() * sizeof(T));
        T* h_in_ptr = (T*)h_in->ptr();
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyAsync(h_in_ptr, in_g->data<T>(), in_g->numel() * sizeof(T), cudaMemcpyDeviceToHost,
                        context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        auto h_idx = memory::Alloc(phi::GPUPinnedPlace(), in_idx->numel() * sizeof(T));
        T* h_idx_ptr = (T*)h_idx->ptr();
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyAsync(h_idx_ptr, in_idx->data<T>(), in_idx->numel() * sizeof(T), cudaMemcpyDeviceToHost,
                        context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        auto h_out = memory::Alloc(phi::GPUPinnedPlace(), out_g->numel() * sizeof(T));
        T* h_out_ptr = (T*)h_out->ptr();
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyAsync(h_out_ptr, out_g->data<T>(), out_g->numel() * sizeof(T), cudaMemcpyDeviceToHost,
                        context.device_context<paddle::platform::CUDADeviceContext>().stream()));
        PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(context.device_context<paddle::platform::CUDADeviceContext>().stream()));

        auto lod_idx = in_idx->lod();
        auto dim_x = in_g->dims();
        auto dim_out = out_g->dims();
        for (size_t i = 0; i < lod_idx[0].size() - 1; i++) {
            int start = lod_idx[0][i];
            int end = lod_idx[0][i+1];
            int batch_idx = i;
            for (int j = start, k = 0; j < end - 1; j++, k++) {
                int idx_start = int(h_idx_ptr[j] + 0.5);
                int idx_end = int(h_idx_ptr[j+1] + 0.5);
                for (int ii = 0; ii < dim_x[2]; ii++) {
                    float tmp_1 = *(h_out_ptr + batch_idx * dim_out[1] * dim_out[2] + k * dim_out[2] + ii);

                    for (int jj = idx_start; jj < idx_end; jj++) {
                        float tmp = *(h_in_ptr + batch_idx * dim_x[1] * dim_x[2] + jj * dim_x[2] + ii);
                        if (fabs(tmp_1 - tmp) >= 0.000001) {
                            abort();
                        }
                    }            
                }
            }
        }
    }
  }
};


}
}



namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    seqpool_scatter,
    ops::SeqpoolScatterCUDAKernel<float>);
REGISTER_OP_CUDA_KERNEL(
    seqpool_scatter_grad,
    ops::SeqpoolScatterGradCUDAKernel<float>);
