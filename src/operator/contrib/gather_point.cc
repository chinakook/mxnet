#include "./gather_point-inl.h"
#include <atomic>

namespace mxnet {

	void gatherpointKernel(int b, int n, int m, const float *inp, const int *idx, float *out)
	{
		for (int i = 0; i < b; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				int a = idx[i*m + j];
				out[(i*m + j) * 3 + 0] = inp[(i*n + a) * 3 + 0];
				out[(i*m + j) * 3 + 1] = inp[(i*n + a) * 3 + 1];
				out[(i*m + j) * 3 + 2] = inp[(i*n + a) * 3 + 2];
			}
		}
	}

	void scatteraddpointKernel(int b, int n, int m, const float *out_g, const int *idx, float *inp_g)
	{
		for (int i = 0; i < b; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				int a = idx[i*m + j];
				inp_g[(i*n + a) * 3 + 0] += out_g[(i*m + j) * 3 + 0];
				inp_g[(i*n + a) * 3 + 1] += out_g[(i*m + j) * 3 + 1];
				inp_g[(i*n + a) * 3 + 2] += out_g[(i*m + j) * 3 + 2];
			}
		}
	}

namespace op {

	template<>
	void GatherPointCompute<cpu>(const nnvm::NodeAttrs& attrs,
		const OpContext& ctx,
		const std::vector<TBlob>& inputs,
		const std::vector<OpReqType>& req,
		const std::vector<TBlob>& outputs) {
		const int B = inputs[0].size(0);
		const int N = inputs[0].size(1);
		const int M = inputs[1].size(1);
		gatherpointKernel(
			B, N, M, inputs[0].dptr<float>(), inputs[1].dptr<int>(), outputs[0].dptr<float>());
	}

	template<>
	void GatherPointGradCompute<cpu>(const nnvm::NodeAttrs& attrs,
		const OpContext& ctx,
		const std::vector<TBlob>& inputs,
		const std::vector<OpReqType>& req,
		const std::vector<TBlob>& outputs) {
		const TBlob &out_grad = inputs[0];
		const TBlob &idx = inputs[1];
		const TBlob &in_grad = outputs[0];

		const int B = in_grad.size(0);
		const int N = in_grad.size(1);
		const int M = idx.size(1);
		scatteraddpointKernel(
			B, N, M, out_grad.dptr<float>(), idx.dptr<int>(), in_grad.dptr<float>());
	}

NNVM_REGISTER_OP(_contrib_GatherPoint)
.describe(R"code(aaaaa)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
	[](const NodeAttrs& attrs) {
	return std::vector<std::string>{"data", "idx"};
})
.set_attr<mxnet::FInferShape>("FInferShape", GatherPointShape)
.set_attr<nnvm::FInferType>("FInferType", GatherPointType)
.set_attr<FCompute>("FCompute<cpu>", GatherPointCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", 
  [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
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

NNVM_REGISTER_OP(_backward_contrib_GatherPoint)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", GatherPointGradCompute<cpu>);

} // namespace op
} // namespace mxnet