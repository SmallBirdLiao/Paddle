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

#pragma once

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/device_context.h"
#endif

namespace paddle {
namespace memory {

/**
 * \brief   Copy memory from one place to another place.
 *
 * \param[in]  DstPlace Destination allocation place (CPU).
 * \param[in]  dst      Destination memory address.
 * \param[in]  SrcPlace Source allocation place (CPU).
 * \param[in]  src      Source memory address.
 * \param[in]  num      memory size in bytes to copy.
 *
 */
template <typename DstPlace, typename SrcPlace>
void Copy(DstPlace, void* dst, SrcPlace, const void* src, size_t num);

/**
 * \brief   Copy memory from one place to another place.
 *
 * \param[in]  DstPlace Destination allocation place (CPU or GPU or XPU or
 * CustomDevice).
 * \param[in]  dst      Destination memory address.
 * \param[in]  SrcPlace Source allocation place (CPU or GPU or XPU or
 * CustomDevice).
 * \param[in]  src      Source memory address.
 * \param[in]  num      memory size in bytes to copy.
 * \param[in]  stream   stream for asynchronously memory copy.
 *
 * \note    For GPU/XPU/CustomDevice memory copy, stream need to be specified
 *          for asynchronously memory copy, and type is restored in the
 *          implementation.
 *
 */
template <typename DstPlace, typename SrcPlace>
void Copy(DstPlace, void* dst, SrcPlace, const void* src, size_t num,
          void* stream);

/**
 * \brief   Copy memory from one place to another place.
 *
 * \param[in]  DstPlace Destination allocation place (CPU or GPU or XPU or
 * CustomDevice).
 * \param[in]  dst      Destination memory address.
 * \param[in]  src      Source memory address.
 * \param[in]  num      memory size in bytes to copy.
 * \param[in]  stream   stream for asynchronously memory copy.
 *
 * \note    For copy constance values from CPU, this function support 
 *          captured by cuda graph.
 *          When its not capturing, this function will call 
 *          Copy(dstplace, dst, src, CPUPlace, num, stream)
 *
 */
template <typename DstPlace>
void CopyConstantFromCPU(DstPlace, void* dst, const void* src, size_t num,
          void* stream);

}  // namespace memory
}  // namespace paddle
