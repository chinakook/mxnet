#include "./gather_point-inl.h"


namespace mxnet {
namespace op {

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
.set_attr<nnvm::FGradient>("FGradient", 
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = nnvm::Node::Create();
    p->attrs.op = nnvm::Op::Get("_backward_GatherPoint");
    p->attrs.name = n->attrs.name + "_backward";
    p->inputs.push_back(ograds[0]);
    p->inputs.push_back(n->inputs[1]);
    p->control_deps.emplace_back(n);
    auto zero = MakeNode("zeros_like", n->attrs.name + "_backward_idx",
                         {n->inputs[1]}, nullptr, &n);

    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{zero, 0, 0});
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol", "points")
.add_argument("idx", "NDArray-or-Symbol", "index");

NNVM_REGISTER_OP(_backward_GatherPoint)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", GatherPointGradCompute<cpu>);

} // namespace op
} // namespace mxnet