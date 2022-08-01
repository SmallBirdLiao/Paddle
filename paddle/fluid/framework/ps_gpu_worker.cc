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

#include <mutex>
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/lodtensor_printer.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#if (defined PADDLE_WITH_NCCL || defined PADDLE_WITH_RCCL) && \
    (defined PADDLE_WITH_PSLIB)
#include "paddle/fluid/platform/cuda_device_guard.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

std::atomic<int> PSGPUWorker::shape_check_count_(16);
std::atomic<bool> PSGPUWorker::shape_check_flag_(false);

void PSGPUWorker::CreateDeviceResource(const ProgramDesc& main_prog) {
  this->HogwildWorker::CreateDeviceResource(main_prog);

  if (build_var_shared_) {
    auto& block = main_prog.Block(0);
    for (auto& var : block.AllVars()) {
      std::string name = var->Name();
      if (!var->Persistable()) {
        build_var_shared_set_.insert(name);
      }
    }
  }

  if (scope_num_ != 1) {
    auto& block = main_prog.Block(0);
    for (int i = 0; i < scope_num_; i++) {
      auto thread_tmp = &thread_scope_->NewScope();
      thread_scope_vec_.push_back(thread_tmp);
    }
    for (auto& scope : thread_scope_vec_) {
      for (auto& var : block.AllVars()) {
        std::string name = var->Name();
        if (!var->Persistable()) {
          auto* ptr = scope->Var(var->Name());
          InitializeVariable(ptr, var->GetType());
        }
      }
    }

    for (auto& op : ops_) {
      op->SetIsRuntimeInferShape(true);
    }
    
    std::vector<std::string> input_names = device_reader_->get_input_var_names();
    std::set<std::string> input_names_set;
    for (auto& i : input_names) {
      input_names_set.insert(i);
    }
    for (auto& scope : thread_scope_vec_) {
      std::vector<Variable*> need_reuse;
      for (auto& var : block.AllVars()) {
        std::string name = var->Name();
        if (!var->Persistable()) {
          if (input_names_set.find(var->Name()) != input_names_set.end()) {
            continue;
          }
          auto* ptr = scope->FindLocalVar(var->Name());
          if (ptr == nullptr) {
            abort();
          }
          need_reuse.push_back(ptr);
        }
      }
      need_reuse_var_vec_[scope] = need_reuse;
    }

    need_reuse_var_.clear();
    for (auto& var : block.AllVars()) {
      std::string name = var->Name();
      if (!var->Persistable()) {
        if (input_names_set.find(var->Name()) != input_names_set.end()) {
          continue;
        }
        auto* ptr = thread_scope_->FindLocalVar(var->Name());
        if (ptr == nullptr) {
          abort();
        }
        need_reuse_var_.push_back(ptr);
      } 
    }
  }
}

void PSGPUWorker::BindingDataFeedMemory() {
  if (scope_num_ == 1) {
    this->HogwildWorker::BindingDataFeedMemory();
  } else {
    for (auto& scope : thread_scope_vec_) {
      device_reader_->AssignFeedVar(*scope);
    }
  }
}

void PSGPUWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  mpi_rank_ = desc.mpi_rank();
  trainer_desc_ = desc;
  for (int i = 0; i < param_.sparse_table_size(); ++i) {
    uint64_t table_id =
        static_cast<uint64_t>(param_.sparse_table(i).table_id());
    TableParameter table = param_.sparse_table(i);
    sparse_key_names_[table_id].resize(table.sparse_key_name_size());
    for (int j = 0; j < table.sparse_key_name_size(); ++j) {
      sparse_key_names_[table_id][j] = table.sparse_key_name(j);
    }
    sparse_value_names_[table_id].resize(table.sparse_value_name_size());
    for (int j = 0; j < table.sparse_value_name_size(); ++j) {
      sparse_value_names_[table_id][j] = table.sparse_value_name(j);
    }
    sparse_grad_names_[table_id].resize(table.sparse_grad_name_size());
    for (int j = 0; j < table.sparse_grad_name_size(); ++j) {
      sparse_grad_names_[table_id][j] = table.sparse_grad_name(j);
    }
    label_var_name_[table_id] = table.label_var_name();
    sparse_push_keys_[table_id] = std::vector<uint64_t>();
  }

  for (int i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_value_names_[table_id].resize(table.dense_value_name_size());
    for (int j = 0; j < table.dense_value_name_size(); ++j) {
      dense_value_names_[table_id][j] = table.dense_value_name(j);
    }
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (int j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }

  skip_ops_.resize(param_.skip_ops_size());
  for (int i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
  for (int i = 0; i < param_.stat_var_names_size(); ++i) {
    stat_var_name_map_[param_.stat_var_names(i)] = 1;
  }

  need_to_push_sparse_ = param_.push_sparse();
  need_to_push_dense_ = param_.push_dense();

  fetch_config_ = desc.fetch_config();
  use_cvm_ = desc.use_cvm();
  // for sparse value accessor, embedding only
  no_cvm_ = desc.no_cvm();
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
  for (int i = 0; i < desc.check_nan_var_names_size(); ++i) {
    check_nan_var_names_.push_back(desc.check_nan_var_names(i));
  }
  copy_table_config_ = desc.copy_table_config();
  for (int i = 0; i < copy_table_config_.src_sparse_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_sparse_tables(i);
    uint64_t dest_table = copy_table_config_.dest_sparse_tables(i);
    VLOG(3) << "copy_sparse_tables_ push back " << src_table << "->"
            << dest_table;
    copy_sparse_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (int i = 0; i < copy_table_config_.src_dense_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_dense_tables(i);
    uint64_t dest_table = copy_table_config_.dest_dense_tables(i);
    VLOG(3) << "copy_dense_tables_ push back " << src_table << "->"
            << dest_table;
    copy_dense_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (auto& m : copy_table_config_.table_denpendency_map()) {
    if (sparse_key_names_.find(m.key()) != sparse_key_names_.end()) {
      // currently only support one dependency
      for (auto& value : m.values()) {
        table_dependency_[m.key()] = value;
      }
    }
  }
}

void PSGPUWorker::SetChannelWriter(ChannelObject<std::string>* queue) {
  writer_.Reset(queue);
}

void PSGPUWorker::PrepareCudaGraph() {
  std::string enable_cuda_graph_capture_attr_name = "enable_cuda_graph_capture";
  op_or_cudagraphs_.reserve(ops_.size());

  static const std::unordered_set<std::string> op_whitelist = {
    "adam",
    "coalesce_tensor",
  };
  // these op can not be captured
  static const std::unordered_set<std::string> op_blacklist = {
    "c_sync_calc_stream",
    "c_allreduce_sum",
    "c_sync_comm_stream",
  };
  // when op is captured, its inputs and outputs and their grads will be never changed
  // so the capture attribute can infect another op whose all inputs and outputs nerver changed
  std::unordered_set<std::string> var_whitelist;
  for (auto& op : ops_) {
    if (op_whitelist.find(op->Type()) != op_whitelist.end() ||
        (op->HasAttr(enable_cuda_graph_capture_attr_name) && op->Attr<int>(enable_cuda_graph_capture_attr_name))) {
      for (auto& input : op->InputVars()) {
        var_whitelist.emplace(input);
        var_whitelist.emplace(framework::GradVarName(input));
      }
      for (auto& output : op->OutputVars(true)) {
        var_whitelist.emplace(output);
        var_whitelist.emplace(framework::GradVarName(output));
      }
    }
  }

  for (auto& op : ops_) {
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (!need_skip) {
      bool need_capture = false;
      // if (op_blacklist.find(op->Type()) == op_blacklist.end()) {
      //   if (op->HasAttr(enable_cuda_graph_capture_attr_name) && op->Attr<int>(enable_cuda_graph_capture_attr_name)) {
      //     need_capture = true;
      //   }
      //   if (!need_capture) {
      //     need_capture = true;
      //     for (auto& input : op->InputVars()) {
      //       if (var_whitelist.find(input) == var_whitelist.end()) {
      //         need_capture = false;
      //         break;
      //       }
      //     }
      //     if (need_capture) {
      //       for (auto& output : op->OutputVars(true)) {
      //         if (var_whitelist.find(output) == var_whitelist.end()) {
      //           need_capture = false;
      //           break;
      //         }
      //       }
      //     }
      //   }
      // }

      if (op_or_cudagraphs_.empty() || op_or_cudagraphs_.back().need_capture != need_capture) {
        op_or_cudagraphs_.emplace_back();
        op_or_cudagraphs_.back().need_capture = need_capture;
      }
      auto& op_or_cuda_graph = op_or_cudagraphs_.back();
      if (need_capture) {
        if (op_or_cuda_graph.name.empty()) {
          op_or_cuda_graph.name = "cuda_graph:";
        } else {
          op_or_cuda_graph.name += ":";
        }
        op_or_cuda_graph.name += op->Type();
      }
      op_or_cuda_graph.ops.emplace_back(op);
    }
  }
}

PSGPUWorker::~PSGPUWorker() {
  stop_token_.store(true);
  for (auto& thread : task_threads_) {
    if (thread.joinable()) {
        thread.join();
    }
  }
}

int PSGPUWorker::OpRunAndShapeCheck(OperatorBase& op,
                                    const Scope& scope,
                                    const platform::Place& place,
                                    size_t op_index, size_t batch_index) {
    if (shape_check_flag_.load()) {
      VLOG(0) << "Begin OpRunAndShapeCheck... "
            << shape_check_count_.load();
      if (shape_check_count_.fetch_sub(1) <= 0) {
        shape_check_flag_ = false;
      }
      // before op run
      InferShapeCheckData check_data;
      auto& pre_dims = check_data.pre_dims;
      auto& pre_lods = check_data.pre_lods;
      auto& after_dims = check_data.after_dims;
      auto& after_lods = check_data.after_lods;
      RuntimeContext ctx(op.Inputs(), op.Outputs(), scope);
      RuntimeInferShapeContext infer_shape_ctx(op, ctx);
      auto outnames = op.Outputs();
      for (auto& var_name_item : outnames) {
        pre_dims.push_back(infer_shape_ctx.GetOutputsDim(var_name_item.first));
        pre_lods.push_back(infer_shape_ctx.GetOutputsLod(var_name_item.first));
      }

      // op run
      op.Run(scope, place);

      // after op run
      for (auto& var_name_item : outnames) {
        after_dims.push_back(infer_shape_ctx.GetOutputsDim(var_name_item.first));
        after_lods.push_back(infer_shape_ctx.GetOutputsLod(var_name_item.first));
      }
      std::string op_name = "unknow_op";
      if (op.Info().HasOpProtoAndChecker()) {
        op_name = op.Info().Proto().type();
      }
      #define SHAPE_CHECK_EQ(__VAL0, __VAL1) \
        PADDLE_ENFORCE_EQ(__VAL0, __VAL1, platform::errors::Fatal( \
                          "Shape check dims/lods error, op name: %s .", op_name))

      SHAPE_CHECK_EQ(pre_dims.size(), after_dims.size());
      for (size_t i = 0; i < pre_dims.size(); i++) {
        SHAPE_CHECK_EQ(pre_dims[i].size(), after_dims[i].size());
        for (size_t j = 0; j < pre_dims[i].size(); j++) {
          SHAPE_CHECK_EQ(pre_dims[i][j], after_dims[i][j]);
        }
      }

      SHAPE_CHECK_EQ(pre_lods.size(), after_lods.size());
      for (size_t i = 0; i < pre_lods.size(); i++) {
        SHAPE_CHECK_EQ(pre_lods[i].size(), after_lods[i].size());
        for (size_t j = 0; j < pre_lods[i].size(); j++) {
          auto& x = pre_lods[i][j];
          auto& y = after_lods[i][j];
          SHAPE_CHECK_EQ(x.size(), y.size());
          for (size_t i = 0; i < x.size(); i++) {
            const auto &x_level = x[i];
            const auto &y_level = y[i];
            SHAPE_CHECK_EQ(x_level.size(), y_level.size());
            for (size_t j = 0; j < x_level.size(); j++) {
              SHAPE_CHECK_EQ(x_level[j], y_level[j]);
            }
          }
        }
      }
      #undef SHAPE_CHECK_EQ
    } else if (batch_index == 1 && build_var_shared_) {
      op.Run(scope, place);
      
      //启发式做变量显存共享
      std::string op_name = "unknow_op";
      if (op.Info().HasOpProtoAndChecker()) {
        op_name = op.Info().Proto().type();
      }
      //先做变量共享
      RuntimeContext ctx(op.Inputs(), op.Outputs(), scope);
      RuntimeInferShapeContext infer_shape_ctx(op, ctx);
      auto outnames = op.Outputs();
      for (auto& var_name_item : outnames) {
        auto first_name = var_name_item.first;
        auto vars = infer_shape_ctx.GetOutputVarPtrs(first_name);
        for (size_t ii = 0; ii < var_name_item.second.size(); ii++) {
          auto second_name = var_name_item.second[ii];
          if (second_name == std::string("@EMPTY@")) {
            break;
          }
          if (build_var_shared_set_.find(second_name) == build_var_shared_set_.end()) {
            continue;
          }
          Variable *cur_var = get<Variable *>(vars[ii]);
          if (cur_var == nullptr) {
            continue;
          }
          if (!cur_var->IsType<LoDTensor>()) {
            continue;
          }
          auto cur_tensor = cur_var->GetMutable<LoDTensor>();
          if (cur_tensor == nullptr || !cur_tensor->initialized()) {
            continue;
          }
          int64_t cur_numel = cur_tensor->numel();

          //int64_t shard_numel = 0;
          std::string shard_name = "";
          Variable *shard_var = nullptr;
          for (auto& iter : build_var_shared_in_shared_) {
            auto tmp_var = scope.FindVar(iter);
            if (tmp_var == nullptr) {
              continue;
            }
            auto tmp_tensor = tmp_var->GetMutable<LoDTensor>();
            if (tmp_tensor == nullptr || !tmp_tensor->initialized()) {
              continue;
            }
            if (tmp_tensor->place() != cur_tensor->place()) {
              continue;
            }
            if (tmp_tensor->dtype() != cur_tensor->dtype()) {
              continue;
            }
            int64_t tmp_numel = tmp_tensor->numel();
            if (tmp_numel == cur_numel) {
              //shard_numel = tmp_numel;
              shard_name = iter;
              shard_var = tmp_var;
              break;
            }
            /*
            if (shard_numel == 0) {
              shard_numel = tmp_numel;
              shard_name = iter;
              shard_var = tmp_var;
            } else {
              if (shard_numel < cur_numel && tmp_numel < cur_numel) {
                if (tmp_numel > shard_numel) {
                  shard_numel = tmp_numel;
                  shard_name = iter;
                  shard_var = tmp_var;
                }
              } else if (shard_numel < cur_numel && tmp_numel >= cur_numel) {
                shard_numel = tmp_numel;
                shard_name = iter;
                shard_var = tmp_var;
              } else {
                if (tmp_numel < shard_numel) {
                  shard_numel = tmp_numel;
                  shard_name = iter;
                  shard_var = tmp_var;
                }
              }
            }
            */
          }
          if (shard_var != nullptr) {
            VLOG(0) << "lxchtestdd  " << shard_name << "  " << second_name << "  "
                                      << (uint64_t)(cur_tensor->data()) << "  " << (uint64_t)(shard_var->GetMutable<LoDTensor>()->data());
            shard_var->GetMutable<LoDTensor>()->ShareBufferWith(*(cur_var->GetMutable<LoDTensor>()));
            build_var_shared_in_shared_.erase(shard_name);
          }
        }
      }


      //后续不用的变量加入到可共享的变量中
      auto iter = build_var_shared_map_.find(op_index);
      if (iter != build_var_shared_map_.end()) {
        for (auto& iter1 : iter->second) {
          build_var_shared_in_shared_.insert(iter1);
        }
      }
    } else {
      op.Run(scope, place);
      /*
      std::string op_name = "unknow_op";
      if (op.Info().HasOpProtoAndChecker()) {
        op_name = op.Info().Proto().type();
      }
      VLOG(0) << "lxchaa  " << "index:" << lxch_index << "  op_name:" << op_name;
      RuntimeContext ctx(op.Inputs(), op.Outputs(), scope);
      RuntimeInferShapeContext infer_shape_ctx(op, ctx);
      auto outnames = op.Outputs();
      for (auto& var_name_item : outnames) {
        auto first_name = var_name_item.first;
        auto dims = infer_shape_ctx.GetOutputsDim(first_name);
        using InferShapeVarPtr = boost::variant<VarDesc *, Variable *>;

        auto vars = infer_shape_ctx.GetOutputVarPtrs(first_name);
        std::vector<std::string> second_names;
        for (auto& nnn : var_name_item.second) {
          if (nnn != std::string("@EMPTY@")) {
            second_names.push_back(nnn);
          }
        }
        if (second_names.size() != dims.size()) {
          abort();
        }
        for (size_t jj = 0; jj < dims.size(); jj++) {
          auto second_name = second_names[jj];
          auto dim = dims[jj];
          size_t dim_size = 1;
          for (int ii = 0; ii < dim.size(); ii++) {
            dim_size *= dim[ii];
          }
          size_t bytes = 0;
          Variable *cur_var = get<Variable *>(vars[jj]);
          if (cur_var != nullptr) {
            if (cur_var->IsType<LoDTensor>()) {
              auto cur_tensor = cur_var->GetMutable<LoDTensor>();
              if (cur_tensor != nullptr && cur_tensor->initialized()) {
                bytes = cur_tensor->capacity();
              }
            }
          }
          VLOG(0) << "lxchbb  " << "index:" << lxch_index << "  op_name:" << op_name
                  << "  biaozhu:" << first_name << "  cur_name:" << second_name
                  << "  size:" << dim_size << "  bytes:" << bytes;
        }
      }
      */
    }
    return 0;
}

void PSGPUWorker::BuildVarShared() {
  //trick处理，如果一个变量，出现在两个算子的output里面，直接设置这个变量不能被共享

  std::set<std::string> output_dup_set;
  //step1 每个变量最后一个op用到的索引值
  std::map<std::string, size_t> last_used_index_map;
  for (size_t ii = 0; ii < ops_.size(); ii++) {
    auto& op = ops_[ii];
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (need_skip) {
      continue;
    }
    std::string op_name = "unknow_op";
    if (op->Info().HasOpProtoAndChecker()) {
      op_name = op->Info().Proto().type();
    }

    auto& inputs = op->Inputs();
    auto& outputs = op->Outputs();
    for (auto& var_name_item : inputs) {
      for (auto& nnn : var_name_item.second) {
        if (nnn == std::string("@EMPTY@")) {
          continue;
        }
        if (op_name == std::string("coalesce_tensor")) {
          build_var_shared_set_.erase(nnn);
          continue;
        }
        if (build_var_shared_set_.find(nnn) != build_var_shared_set_.end()) {
          last_used_index_map[nnn] = ii;
        }
      }
    }
    for (auto& var_name_item : outputs) {
      for (auto& nnn : var_name_item.second) {
        if (nnn == std::string("@EMPTY@")) {
          continue;
        }
        if (op_name == std::string("coalesce_tensor")) {
          build_var_shared_set_.erase(nnn);
          continue;
        }

        if (output_dup_set.find(nnn) != output_dup_set.end()) {
          build_var_shared_set_.erase(nnn);
          VLOG(0) << "lxchtestff " << nnn;
          continue;
        }
        output_dup_set.insert(nnn);

        if (build_var_shared_set_.find(nnn) != build_var_shared_set_.end()) {
          last_used_index_map[nnn] = ii;
        }
      }
    }
  }
  
  //step2 每个算子结束后可以共享的数据
  for (size_t ii = 0; ii < ops_.size(); ii++) {
    auto& op = ops_[ii];
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (need_skip) {
      continue;
    }
    auto& inputs = op->Inputs();
    auto& outputs = op->Outputs();
    for (auto& var_name_item : inputs) {
      for (auto& nnn : var_name_item.second) {
        if (nnn == std::string("@EMPTY@")) {
          continue;
        }
        if (build_var_shared_set_.find(nnn) != build_var_shared_set_.end()) {
          if (last_used_index_map[nnn] <= ii) {
            build_var_shared_map_[ii].push_back(nnn);
          }
        }
      }
    }
    for (auto& var_name_item : outputs) {
      for (auto& nnn : var_name_item.second) {
        if (nnn == std::string("@EMPTY@")) {
          continue;
        }
        if (build_var_shared_set_.find(nnn) != build_var_shared_set_.end()) {
          if (last_used_index_map[nnn] <= ii) {
            build_var_shared_map_[ii].push_back(nnn);
          }
        }
      }
    }
  }
}

void PSGPUWorker::TrainFiles() {
  VLOG(0) << "Begin to train files";
  platform::SetNumThreads(1);
  platform::Timer timeline;
  timeline.Start();

  if (build_var_shared_) {
    BuildVarShared();
  }

    struct RunOpInfo {
    RunOpInfo() {
      times = 0;
      costs = 0;
      pre_costs = 0;
      run_costs = 0;
      after_costs = 0;
      other_times = 0;
      first_costs = 0;
      time_1 = 0;
      time_2 = 0;
      time_3 = 0;
      time_4 = 0;
      time_5 = 0;
      time_6 = 0;
    }
    uint64_t other_times;
    uint64_t times;
    uint64_t costs;
    uint64_t pre_costs;
    uint64_t run_costs;
    uint64_t after_costs;
    uint64_t first_costs;
    uint64_t time_1;
    uint64_t time_2;
    uint64_t time_3;
    uint64_t time_4;
    uint64_t time_5;
    uint64_t time_6;
  };
  std::vector<RunOpInfo> op_time_info;
  op_time_info.resize(ops_.size());

  int total_ins_num = 0;

  // how to accumulate fetched values here
  device_reader_->Start();
  int cur_batch;
  int batch_cnt = 0;

  int graph_batch_size = 0;

  platform::SetDeviceId(place_.GetDeviceId());

  // async infershape
  pack_is_end_.store(false);
  if (scope_num_ != 1) {
    for (size_t i = 0; i < thread_scope_vec_.size(); i++) {
      TaskData task;
      task.scope = thread_scope_vec_[i];
      free_task_queue_.Push(task);
    }
    thread_count_.store(task_threads_num_);
    task_threads_.reserve(task_threads_num_);
    for (int i = 0; i < task_threads_num_; i++) {
      task_threads_.emplace_back(std::thread([this]() -> void {
        while (true) {
          auto pack = device_reader_->get_pack(nullptr);
          if (pack == nullptr) {
            int thread_num = thread_count_.fetch_sub(1);
            if (thread_num == 1) {
              pack_is_end_.store(true);
            }
            return;
          }
          auto task = free_task_queue_.Pop();
          task.pack = pack;
          task.ins_num = pack->ins_num();
          device_reader_->PackToScope(task.pack, task.scope);
          for (size_t ii = 0; ii < ops_.size(); ii++) {
            auto& op = ops_[ii];
            bool need_skip = false;
            for (auto t = 0u; t < skip_ops_.size(); ++t) {
              if (op->Type().find(skip_ops_[t]) != std::string::npos) {
                need_skip = true;
                break;
              }
            }
            if (!need_skip) {
              op->RuntimeInferShape(*task.scope);
            }
          }
          using_task_queue_.Push(task);
        }
      }));
    }
  }
  
  size_t batch_index = 0;
  while (true) {
    auto thread_scope = thread_scope_;
    TaskData cur_task;
    if (scope_num_ == 1) {
      cur_batch = device_reader_->Next();
    } else {
      while (true) {
        if (using_task_queue_.Size() != 0) {
          cur_task = using_task_queue_.Pop();
          cur_batch = cur_task.ins_num;
          break;
        }
        bool is_end = pack_is_end_.load();
        if (is_end) {
          if (using_task_queue_.Size() == 0) {
            cur_batch = 0;
            break;
          }
        }
        std::this_thread::sleep_for(
          std::chrono::microseconds(200));
      }
      thread_scope = cur_task.scope;
      std::vector<Variable*>& cur_scope_vars = need_reuse_var_vec_[thread_scope];
      for (size_t iii = 0; iii < need_reuse_var_.size(); iii++) {
        Variable* l_v = cur_scope_vars[iii];
        Variable* r_v = need_reuse_var_[iii];
        if (l_v->IsType<LoDTensor>()) {
          l_v->GetMutable<LoDTensor>()->ShareBufferWith(*(r_v->GetMutable<LoDTensor>()));
        }
      }
    }

    if (cur_batch <= 0) {
      break;
    }

    total_ins_num += cur_batch;
    batch_index++;

    if (op_or_cudagraphs_.empty()) {
      // first batch we run original ops to check whethere the tensors has lod
      for (size_t ii = 0; ii < ops_.size(); ii++) {
        auto& op = ops_[ii];
        uint64_t lxch_op_1 = platform::Timer::lxch_get_base_time();
        bool need_skip = false;
        for (auto t = 0u; t < skip_ops_.size(); ++t) {
          if (op->Type().find(skip_ops_[t]) != std::string::npos) {
            need_skip = true;
            break;
          }
        }
        if (!need_skip) {
          OpRunAndShapeCheck(*op, *thread_scope, place_, ii, batch_index);
          uint64_t lxch_op_2 = platform::Timer::lxch_get_base_time();
          if (op->lxch_time_1 >= lxch_op_1) {
            op_time_info[ii].pre_costs += op->lxch_time_1 - lxch_op_1;
            op_time_info[ii].after_costs += lxch_op_2 - op->lxch_time_2;
            op_time_info[ii].run_costs += op->lxch_time_2 - op->lxch_time_1;

            op_time_info[ii].time_1 += op->lxch_time_3 - lxch_op_1;
            op_time_info[ii].time_2 += op->lxch_time_4 - op->lxch_time_3;
            op_time_info[ii].time_3 += op->lxch_time_5 - op->lxch_time_4;
            op_time_info[ii].time_4 += op->lxch_time_6 - op->lxch_time_5;
            op_time_info[ii].time_5 += op->lxch_time_7 - op->lxch_time_6;
            op_time_info[ii].time_6 += op->lxch_time_1 - op->lxch_time_7;
          } else {
            op_time_info[ii].other_times++;
          }
          if (op_time_info[ii].times == 0) {
            op_time_info[ii].first_costs = op->lxch_time_1 - lxch_op_1;
          }
          op_time_info[ii].times++;
          op_time_info[ii].costs += lxch_op_2 - lxch_op_1;
        }
      }
      graph_batch_size = cur_batch;
//      PrepareCudaGraph();
    } else if (graph_batch_size != cur_batch || batch_cnt <= thread_id_) {
      // when batch_size changed, run original ops
      for (auto& op : ops_) {
        bool need_skip = false;
        for (auto t = 0u; t < skip_ops_.size(); ++t) {
          if (op->Type().find(skip_ops_[t]) != std::string::npos) {
            need_skip = true;
            break;
          }
        }
        if (!need_skip) {
          OpRunAndShapeCheck(*op, *thread_scope, place_);
        }
      }
    } else {
      // secend batch we capture the cudagraph
      for (auto& op_or_cuda_graph : op_or_cudagraphs_) {
        if (op_or_cuda_graph.need_capture) {
          if (op_or_cuda_graph.cudagraph == nullptr) {
            static std::mutex _capture_mutex;
            std::lock_guard<std::mutex> lock(_capture_mutex);
            platform::BeginCUDAGraphCapture(place_, cudaStreamCaptureModeThreadLocal);
            for (auto& op : op_or_cuda_graph.ops) {
              OpRunAndShapeCheck(*op, *thread_scope, place_);
            }
            op_or_cuda_graph.cudagraph = platform::EndCUDAGraphCapture();
          }

          platform::RecordEvent op_type_record_event(
              op_or_cuda_graph.name, platform::TracerEventType::Operator, 1);
          op_or_cuda_graph.cudagraph->Replay();
        } else {
          for (auto& op : op_or_cuda_graph.ops) {
            OpRunAndShapeCheck(*op, *thread_scope, place_);
          }
        }
      }
    }
    if (need_dump_field_) {
      DumpField(*thread_scope, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && thread_id_ == 0) {
      DumpParam(*thread_scope, batch_cnt);
    }

    for (std::string& var_name : check_nan_var_names_) {
      Variable* var = thread_scope->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (tensor == nullptr || !tensor->IsInitialized()) {
        continue;
      }
      if (framework::TensorContainsInf(*tensor) ||
          framework::TensorContainsNAN(*tensor)) {
        static std::mutex mutex;
        {
          std::lock_guard<std::mutex> lock(mutex);
          VLOG(0) << "worker " << thread_id_ << ": " << var_name
                  << " cantains inf or nan";
          auto all_vars = thread_scope->LocalVarNames();
          std::stringstream ss;
          ss << "====== worker " << thread_id_ << "======\n";
          for (auto& local_var : all_vars) {
            platform::PrintVar(thread_scope, local_var, local_var, &ss);
            ss << "\n";
          }
          std::cout << ss.str() << std::endl;
          VLOG(0) << "worker " << thread_id_ << "print nan var done....";
        }
        sleep(600);
        exit(-1);
      }
    }

    dev_ctx_->Wait();
    PrintFetchVars();
    thread_scope->DropKids();
    ++batch_cnt;

    if (scope_num_ != 1) {
      device_reader_->get_pack(cur_task.pack);
      free_task_queue_.Push(cur_task);
      std::vector<Variable*>& cur_scope_vars = need_reuse_var_vec_[thread_scope];
      for (size_t iii = 0; iii < need_reuse_var_.size(); iii++) {
        Variable* l_v = cur_scope_vars[iii];
        Variable* r_v = need_reuse_var_[iii];
        if (l_v->IsType<LoDTensor>()) {
          r_v->GetMutable<LoDTensor>()->ShareBufferWith(*(l_v->GetMutable<LoDTensor>()));
        }
      }
      
    }
  }
  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }
  timeline.Pause();
  VLOG(0) << "GpuPs worker " << thread_id_ << " train cost "
          << timeline.ElapsedSec() << " seconds, ins_num: " << total_ins_num;

  if (thread_id_ == 0) {
    std::map<std::string, RunOpInfo> lxch_reduce_info;
    for (size_t ii = 0; ii < op_time_info.size(); ii++) {
      std::string op_name = ops_[ii]->Type();
      if (lxch_reduce_info.find(op_name) == lxch_reduce_info.end()) {
        lxch_reduce_info[op_name] = op_time_info[ii];
      } else {
        lxch_reduce_info[op_name].times += op_time_info[ii].times;
        lxch_reduce_info[op_name].costs += op_time_info[ii].costs;
        lxch_reduce_info[op_name].pre_costs += op_time_info[ii].pre_costs;
        lxch_reduce_info[op_name].run_costs += op_time_info[ii].run_costs;
        lxch_reduce_info[op_name].after_costs += op_time_info[ii].after_costs;
        lxch_reduce_info[op_name].other_times += op_time_info[ii].other_times;
        lxch_reduce_info[op_name].first_costs += op_time_info[ii].first_costs;
        lxch_reduce_info[op_name].time_1 += op_time_info[ii].time_1;
        lxch_reduce_info[op_name].time_2 += op_time_info[ii].time_2;
        lxch_reduce_info[op_name].time_3 += op_time_info[ii].time_3;
        lxch_reduce_info[op_name].time_4 += op_time_info[ii].time_4;
        lxch_reduce_info[op_name].time_5 += op_time_info[ii].time_5;
        lxch_reduce_info[op_name].time_6 += op_time_info[ii].time_6;
      }
    }

    for (auto iter = lxch_reduce_info.begin(); iter != lxch_reduce_info.end(); iter++) {
            LOG(ERROR) << "liaoxiaochao-run-detail op["
                 << iter->first << "] times[" << iter->second.times
                 << "] all_costs[" << iter->second.costs << "] pre_costs["
                 << iter->second.pre_costs << "] run_costs["
                 << iter->second.run_costs << "] after_costs["
                 << iter->second.after_costs << "] other_times["
                 << iter->second.other_times << "] first_costs["
                 << iter->second.first_costs << "] time_1_costs["
                 << iter->second.time_1 << "] time_2_costs["
                 << iter->second.time_2 << "] time_3_costs["
                 << iter->second.time_3 << "] time_4_costs["
                 << iter->second.time_4 << "] time_5_costs["
                 << iter->second.time_5 << "] time_6_costs["
                 << iter->second.time_6 << "]";
    }
  }
  return;
}

void PSGPUWorker::TrainFilesWithProfiler() {
  platform::SetNumThreads(1);
  VLOG(0) << "Begin to train files with profiler";
  device_reader_->Start();
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (!need_skip) {
      op_name.push_back(op->Type());
    }
  }

  VLOG(3) << "op name size: " << op_name.size();
  op_total_time.resize(op_name.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  int total_ins_num = 0;
  int cur_batch;
  timeline.Start();
  platform::SetDeviceId(place_.GetDeviceId());
  while ((cur_batch = device_reader_->Next()) > 0) {
    total_ins_num += cur_batch;
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    int run_op_idx = 0;
    dev_ctx_->Wait();
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        timeline.Start();
        VLOG(3) << "Going to run op " << op_name[run_op_idx];
        op->Run(*thread_scope_, place_);
        dev_ctx_->Wait();
        VLOG(3) << "Op " << op_name[run_op_idx] << " Finished";
        timeline.Pause();
        op_total_time[run_op_idx++] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }
    timeline.Start();
    PrintFetchVars();
    thread_scope_->DropKids();
    dev_ctx_->Wait();
    timeline.Pause();
    total_time += timeline.ElapsedSec();
    timeline.Start();
  }
  VLOG(0) << "GpuPs worker " << thread_id_ << " train cost " << total_time
          << " seconds, ins_num: " << total_ins_num;
  for (size_t i = 0; i < op_name.size(); ++i) {
    VLOG(0) << "card:" << thread_id_ << ", op: " << op_name[i]
            << ", mean time: " << op_total_time[i] / total_ins_num
            << "s, totol time:" << op_total_time[i] << "sec";
  }
  VLOG(0) << "card: " << thread_id_ << " read time: " << read_time
          << ", percent: " << read_time / total_time * 100;
  return;
}

void PSGPUWorker::ResetStat() {
  total_time_ = 0;
  read_time_ = 0;
  pack_time_ = 0;
  pull_sparse_local_time_ = 0;
  op_all_time_ = 0;
  xpu_op_time_ = 0;
  xpu_wait_time_ = 0;
  cpu_op_time_ = 0;
  collect_label_time_ = 0;
  fill_sparse_time_ = 0;
  push_sparse_time_ = 0;
  gpu_2_cpu_time_ = 0;
  cpu_2_gpu_time_ = 0;
  total_inst_ = 0;
}

void PSGPUWorker::ProduceTasks() { return; }

}  // end namespace framework
}  // end namespace paddle
#endif
