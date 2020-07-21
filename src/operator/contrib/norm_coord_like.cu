#include <mshadow/tensor.h>
#include "./norm_coord_like-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_norm_coord_like)
.set_attr<FCompute>("FCompute<gpu>", NormCoordLikeCompute<gpu>);

}  // namespace op
}  // namespace mxnet