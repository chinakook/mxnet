#include "./farthest_point_sampling-inl.h"

namespace mxnet {

	void farthestpointsamplingKernel(int b, int n, int m, const float *dataset, float *temp, int *idxs)
	{
		for (int i = 0; i < b; ++i)
		{
			int old = 0;
			idxs[i * m + 0] = old;
			for (int j = 0; j < n; ++j)
			{
				temp[j] = std::numeric_limits<float>::max();
			}

			for (int k = 1; k < m; ++k)
			{
				float x1 = dataset[(i * n + old) * 3 + 0];
				float y1 = dataset[(i * n + old) * 3 + 1];
				float z1 = dataset[(i * n + old) * 3 + 2];

				float max = -1;
				int jmax = 0;
				for (int j = 0; j < n; ++j)
				{
					float x0 = dataset[(i * n + j) * 3 + 0];
					float y0 = dataset[(i * n + j) * 3 + 1];
					float z0 = dataset[(i * n + j) * 3 + 2];
					float d = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1);

					temp[j] = std::min(temp[j], d);
					if (temp[j] > max)
					{
						max = temp[j];
						jmax = j;
					}
				}
				old = jmax;
				idxs[i * m + k] = old;
			}
		}
	}

	namespace op
	{

	template<>
	void FarthestPointSamplingCompute<cpu>(const nnvm::NodeAttrs& attrs,
		const OpContext& ctx,
		const std::vector<TBlob>& inputs,
		const std::vector<OpReqType>& req,
		const std::vector<TBlob>& outputs) {
		using namespace mshadow;
		CHECK_GT(ctx.requested.size(), 0);
		const int B = inputs[0].size(0);
		const int N = inputs[0].size(1);
		const FarthestPointSamplingParam& param = nnvm::get<FarthestPointSamplingParam>(attrs.parsed);

		auto *stream = ctx.get_stream<cpu>();
		auto tmp_shape = mshadow::Shape2(32, N);

		Tensor<cpu, 2, float> tmp = ctx.requested[0].get_space_typed<cpu, 2, float>(tmp_shape, stream);
		//Fill<false>(stream, TBlob(reinterpret_cast<nnvm::dim_t*>(tmp.dptr_), tmp_shape, cpu::kDevMask), kWriteTo, 1e10f);

		farthestpointsamplingKernel(
			B, N, param.npoints, inputs[0].dptr<float>(), tmp.dptr_, outputs[0].dptr<int>());
	}

	template<>
	void FarthestPointSamplingGradCompute<cpu>(const nnvm::NodeAttrs& attrs,
		const OpContext& ctx,
		const std::vector<TBlob>& inputs,
		const std::vector<OpReqType>& req,
		const std::vector<TBlob>& outputs) {

	}

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
.set_attr<mxnet::FInferShape>("FInferShape", FarthestPointSamplingShape)
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