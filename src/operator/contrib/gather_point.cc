#include "./gather_point-inl.h"


namespace mxnet {
namespace op {

	struct GatherPointGrad {
		const char *op_name;
		std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
			const std::vector<nnvm::NodeEntry>& ograds) const {
			std::vector<nnvm::NodeEntry> heads;
			heads.push_back(ograds[0]);
			heads.push_back(n->inputs[1]);

			return MakeGradNode(op_name, n, heads, n->attrs.dict);
		}
	};

NNVM_REGISTER_OP(_contrib_GatherPoint)
.describe(R"code(aaaaa)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
	[](const NodeAttrs& attrs) {
	return std::vector<std::string>{"data", "idx"};
})
.set_attr<nnvm::FInferShape>("FInferShape", GatherPointShape)
.set_attr<nnvm::FInferType>("FInferType", GatherPointType)
.set_attr<FCompute>("FCompute<cpu>", GatherPointCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", GatherPointGrad{"_backward_GatherPoint" })
.add_argument("data", "NDArray-or-Symbol", "points")
.add_argument("idx", "NDArray-or-Symbol", "index");

NNVM_REGISTER_OP(_backward_GatherPoint)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", GatherPointGradCompute<cpu>);

} // namespace op
} // namespace mxnet