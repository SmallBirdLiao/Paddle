

namespace paddle {
namespace framework {

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


}
}