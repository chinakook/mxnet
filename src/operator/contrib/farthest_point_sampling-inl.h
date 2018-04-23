#ifndef MXNET_OPERATOR_FARTHEST_POINT_SAMPLING_INL_H_
#define MXNET_OPERATOR_FARTHEST_POINT_SAMPLING_INL_H_

#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

    struct FarthestPointSamplingParam : public dmlc::Parameter<FarthestPointSamplingParam>
    {
        int npoints;
        DMLC_DECLARE_PARAMETER(FarthestPointSamplingParam) {
            DMLC_DECLARE_FIELD(npoints)
                .describe("sampling points number.");
        }
    };

	inline bool FarthestPointSamplingShape(const nnvm::NodeAttrs& attrs,
		std::vector<TShape>* in_attrs,
		std::vector<TShape>* out_attrs) {
		CHECK_EQ(in_attrs->size(), 1U);
		CHECK_EQ(out_attrs->size(), 1U);

		const TShape& dshape = (*in_attrs)[0];

		if (shape_is_none(dshape)) return false;

        const FarthestPointSamplingParam& param = nnvm::get<FarthestPointSamplingParam>(attrs.parsed);

		TShape oshape(2);
		oshape[0] = dshape[0];
		oshape[1] = param.npoints;
		//oshape[2] = 3;

		SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
		return true;
	}

	inline bool FarthestPointSamplingType(const nnvm::NodeAttrs& attrs,
		std::vector<int>* in_attrs,
		std::vector<int>* out_attrs) {
		CHECK_EQ(in_attrs->size(), 1U);
		CHECK_EQ(out_attrs->size(), 1U);

        TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
		TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt32);
		return true;
	}



	template<typename xpu>
	void FarthestPointSamplingCompute(const nnvm::NodeAttrs& attrs,
		const OpContext& ctx, const std::vector<TBlob>& inputs,
		const std::vector<OpReqType>& req,
		const std::vector<TBlob>& outputs) {
// 	const int B = inputs[0].size(0);
// 	const int N = inputs[0].size(1);
//     const FarthestPointSamplingParam& param = nnvm::get<FarthestPointSamplingParam>(attrs.parsed);

// 	auto *stream = ctx.get_stream<gpu>();
// 	auto s = mshadow::Stream<mxnet::gpu>::GetStream(stream);
// using namespace mshadow;
//     Shape<2> shape;
//     shape.shape_[0] = B;
//     shape.shape_[1] = N;
//     printf("FPS_BEGIN\n");
//     mshadow::Shape2(B, N);
//     Tensor<gpu, 2, float> tmp = ctx.requested[0].get_space_typed<gpu, 2, float>(shape, stream);
		//LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
	}

	template<typename xpu>
	void FarthestPointSamplingGradCompute(const nnvm::NodeAttrs& attrs,
		const OpContext& ctx, const std::vector<TBlob>& inputs,
		const std::vector<OpReqType>& req,
		const std::vector<TBlob>& outputs) {

		//LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
	}

} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_FARTHEST_POINT_SAMPLING_INL_H_