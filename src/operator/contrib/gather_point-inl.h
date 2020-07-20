#ifndef MXNET_OPERATOR_GATHER_POINT_INL_H_
#define MXNET_OPERATOR_GATHER_POINT_INL_H_

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

	inline bool GatherPointShape(const nnvm::NodeAttrs& attrs,
		std::vector<TShape>* in_attrs,
		std::vector<TShape>* out_attrs) {
		CHECK_EQ(in_attrs->size(), 2U);
		CHECK_EQ(out_attrs->size(), 1U);

		const TShape& dshape = (*in_attrs)[0];
		const TShape& ishape = (*in_attrs)[1];

		if (shape_is_none(dshape) || shape_is_none(ishape)) return false;

		CHECK_EQ(ishape.ndim(), 2)
			<< "gather_point requires index tensor to have 2 dimensions";

		TShape oshape({ishape[0], ishape[1], 3});
		//oshape[0] = ishape[0];
		//oshape[1] = ishape[1];
		//oshape[2] = 3;

		SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
		return true;
	}

	inline bool GatherPointType(const nnvm::NodeAttrs& attrs,
		std::vector<int>* in_attrs,
		std::vector<int>* out_attrs) {
		CHECK_EQ(in_attrs->size(), 2U);
		CHECK_EQ(out_attrs->size(), 1U);

		TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
		TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
		return true;
	}



	template<typename xpu>
	void GatherPointCompute(const nnvm::NodeAttrs& attrs,
		const OpContext& ctx, const std::vector<TBlob>& inputs,
		const std::vector<OpReqType>& req,
		const std::vector<TBlob>& outputs) {

		//LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
	}

	template<typename xpu>
	void GatherPointGradCompute(const nnvm::NodeAttrs& attrs,
		const OpContext& ctx, const std::vector<TBlob>& inputs,
		const std::vector<OpReqType>& req,
		const std::vector<TBlob>& outputs) {

		//LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
	}

} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_GATHER_POINT_INL_H_