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
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class SeqpoolScatterOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
      OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SeqpoolScatter");
      OP_INOUT_CHECK(ctx->HasInput("Idx"), "Input", "Idx", "SeqpoolScatter");
      OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SeqpoolScatter");
      if (!ctx->IsRuntime()) {
        ctx->SetLoDLevel("Out", 0);
      }
      //先只支持sum-pooling
      auto pooltype = ctx->Attrs().Get<std::string>("pooltype");
      PADDLE_ENFORCE_EQ(pooltype, "SUM",
                      platform::errors::InvalidArgument(
                          "only support SUM pooling. But "
                          "received: pooltype %s",
                          pooltype.c_str()));

      //特化判断逻辑
      auto x_dims = ctx->GetInputDim("X");
      auto idx_dims = ctx->GetInputDim("Idx");
      
      PADDLE_ENFORCE_EQ(x_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "The rank of Input(X) must be 3. But "
                          "received: Input(X) rank %u",
                          x_dims.size()));
      PADDLE_ENFORCE_EQ(idx_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The rank of Input(Idx) must be 2. But "
                          "received: Input(Idx) rank %u",
                          idx_dims.size()));
      if (!ctx->IsRuntime()) {
        auto x_lod_level = ctx->GetLoDLevel("X");
        PADDLE_ENFORCE_EQ(x_lod_level, 0,
                      platform::errors::InvalidArgument(
                          "The value of Input(X).lod_level must be 0. But "
                          "received: Input(X).lod_level is %u",
                          x_lod_level));
        auto idx_lod_level = ctx->GetLoDLevel("Idx");
        PADDLE_ENFORCE_LE(idx_lod_level, 1,
                      platform::errors::InvalidArgument(
                          "The value of Input(idx).lod_level must be 0 or 1. But "
                          "received: Input(idx).lod_level is %u",
                          idx_lod_level));
        if (idx_lod_level == 1) {
          PADDLE_ENFORCE_EQ(idx_dims[1], 1,
                          platform::errors::InvalidArgument(
                              "The value of Input(idx).dim[1] must be 1. But "
                              "received: Input(idx).dim[1] is %u",
                              idx_dims[1]));
        }
      }

      ctx->SetOutputDim("Out", ctx->GetInputDim("X"));

  }
};

class SeqpoolScatterOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
      AddInput("X", "(LoDTensor) The variable-length input of SeqpoolScatterOp");
      AddInput("Idx", "(LoDTensor) The LodInfo input of SeqpoolScatterOp");
      AddOutput("Out", "(Tensor) The output of SequencePoolOp does not contain LoD information.");
      AddAttr<std::string>(
        "pooltype",
        "(string, default 'SUM') the pooling pooltype of SequencePoolOp.")
        .SetDefault("SUM")
        .InEnum({"AVERAGE", "SUM", "SQRT", "LAST", "FIRST", "MAX"});
      AddAttr<float>("pad_value",
                   "(float, default 0.0) The value to pad for empty sequence.")
        .SetDefault(0.0);
      AddComment(R"DOC()DOC");
  }
};

class SeqpoolScatterGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SeqpoolScatterGrad");
    OP_INOUT_CHECK(ctx->HasInput("Idx"), "Input", "Idx", "SeqpoolScatterGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "SeqpoolScatterGrad");

    ctx->ShareDim("X", /*->*/ framework::GradVarName("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    return;
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class SeqpoolScatterGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("seqpool_scatter_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("Idx", this->Input("Idx"));
    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op_desc_ptr->SetAttrMap(this->Attrs());
    return;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SeqpoolScatterGradOpNoNeedBufferVarsInferer,
                                    "X", "Idx");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(seqpool_scatter, ops::SeqpoolScatterOp, ops::SeqpoolScatterOpMaker,
                  ops::SeqpoolScatterGradOpMaker<paddle::framework::OpDesc>,
                  ops::SeqpoolScatterGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(seqpool_scatter_grad, ops::SeqpoolScatterGradOp,
                  ops::SeqpoolScatterGradOpNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(
    seqpool_scatter,
    ops::SeqpoolScatterCPUKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SeqpoolScatterCPUKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    seqpool_scatter_grad,
    ops::SeqpoolScatterGradCPUKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SeqpoolScatterGradCPUKernel<paddle::platform::CPUDeviceContext, double>);

