#include "./farthest_point_sampling-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(FarthestPointSamplingParam);

NNVM_REGISTER_OP(_contrib_FarthestPointSampling)
.describe(R"code(aaaaa)code" ADD_FILELINE)
.set_attr_parser(ParamParser<FarthestPointSamplingParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
	[](const NodeAttrs& attrs) {
	return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FInferShape>("FInferShape", FarthestPointSamplingShape)
.set_attr<nnvm::FInferType>("FInferType", FarthestPointSamplingType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", FarthestPointSamplingCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",ElemwiseGradUseNone{"_contrib_backward_FarthestPointSampling"})
.add_argument("data", "NDArray-or-Symbol", "points")
.add_arguments(FarthestPointSamplingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_FarthestPointSampling)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", FarthestPointSamplingGradCompute<cpu>);

} // namespace op
} // namespace mxnet