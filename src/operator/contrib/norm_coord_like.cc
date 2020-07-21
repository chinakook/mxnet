#include "./norm_coord_like-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NormCoordLikeParam);

NNVM_REGISTER_OP(_contrib_norm_coord_like)
.describe("Return evenly spaced numbers over a specified interval. Similar to Numpy")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NormCoordLikeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NormCoordLikeShape)
.set_attr<nnvm::FInferType>("FInferType", NormCoordLikeType)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 0); })
.set_attr<FCompute>("FCompute<cpu>", NormCoordLikeCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input")
.add_arguments(NormCoordLikeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet