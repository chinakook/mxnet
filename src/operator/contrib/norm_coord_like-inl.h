#ifndef MXNET_OPERATOR_NORM_COORD_LIKE_INL_H_
#define MXNET_OPERATOR_NORM_COORD_LIKE_INL_H_

#include <mxnet/base.h>
#include <dmlc/parameter.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/imperative.h>
#include <vector>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../../api/operator/op_utils.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct NormCoordLikeParam : public dmlc::Parameter<NormCoordLikeParam> {
    std::string ctx;
    int dtype;
    DMLC_DECLARE_PARAMETER(NormCoordLikeParam) {
        DMLC_DECLARE_FIELD(ctx)
            .set_default("")
            .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
                "Only used for imperative calls.");
        DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
            .add_enum("float32", mshadow::kFloat32)
            .add_enum("float64", mshadow::kFloat64)
            .describe("Data type.");
    }
};

inline bool NormCoordLikeShape(const nnvm::NodeAttrs& attrs,
    mxnet::ShapeVector* in_attrs,
    mxnet::ShapeVector* out_attrs) {
    // const NormCoordLikeParam& param = nnvm::get<NormCoordLikeParam>(attrs.parsed);
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);
    const TShape& dshape = (*in_attrs)[0];
    TShape oshape({dshape[0], 2, dshape[2], dshape[3]});
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
    return true;
}

inline bool NormCoordLikeType(const nnvm::NodeAttrs& attrs,
    std::vector<int>* in_attrs,
    std::vector<int>* out_attrs) {
    const NormCoordLikeParam& param = nnvm::get<NormCoordLikeParam>(attrs.parsed);
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);

    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
    // TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    return true;
}

struct repeat_x_fwd {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, index_t repeat, DType start, DType step,
                                  int req, DType* out) {
    KERNEL_ASSIGN(out[i], req, start + (i % repeat) * step);
  }
};

struct repeat_y_fwd {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, index_t repeat, DType start, DType step,
                                  int req, DType* out) {
    KERNEL_ASSIGN(out[i], req, start + (i/repeat) * step);
  }
};

template<typename xpu>
void NormCoordLikeCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
    using namespace mxnet_op;
    Stream<xpu>* s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        size_t one_channel_size = outputs[0].size(2) * outputs[0].size(3);
        index_t repeat = outputs[0].size(3);
        int step_num_x = outputs[0].size(3) - 1;
        double step_x = (1. - (-1.)) / step_num_x;
        Kernel<repeat_x_fwd, xpu>::Launch(s,
                                          one_channel_size,
                                          static_cast<int>(repeat),
                                          static_cast<DType>(-1.),
                                          static_cast<DType>(step_x),
                                          req[0],
                                          outputs[0].dptr<DType>());
    });

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        size_t one_channel_size = outputs[0].size(2) * outputs[0].size(3);
        index_t repeat = outputs[0].size(3);
        int step_num_y = outputs[0].size(2) - 1;
        double step_y = (1. - (-1.)) / step_num_y;
        Kernel<repeat_y_fwd, xpu>::Launch(s,
                                          one_channel_size,
                                          static_cast<int>(repeat),
                                          static_cast<DType>(-1.),
                                          static_cast<DType>(step_y),
                                          req[0],
                                          outputs[0].dptr<DType>() + one_channel_size);
        });
}

} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_NORM_COORD_LIKE_INL_H_
