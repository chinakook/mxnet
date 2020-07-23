#include "./gather_point-inl.h"
#include <mshadow/tensor.h>

namespace mxnet {
namespace cuda {

__global__ void gatherpointKernel(int b,int n,int m
    ,const float * __restrict__ inp,const int * __restrict__ idx, float * __restrict__ out){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
			int a=idx[i*m+j];
			out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
			out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
			out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
		}
    }
}

__global__ void scatteraddpointKernel(int b,int n,int m
    ,const float * __restrict__ out_g,const int * __restrict__ idx, float * __restrict__ inp_g){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
			int a=idx[i*m+j];
			atomicAdd(&inp_g[(i*n+a)*3+0],out_g[(i*m+j)*3+0]);
			atomicAdd(&inp_g[(i*n+a)*3+1],out_g[(i*m+j)*3+1]);
			atomicAdd(&inp_g[(i*n+a)*3+2],out_g[(i*m+j)*3+2]);
		}
    }
}

} // namespace cuda

namespace op {

template<>
void GatherPointCompute<gpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
						const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
	const int B = inputs[0].size(0);
	const int N = inputs[0].size(1);
	const int M = inputs[1].size(1);
	auto *stream = ctx.get_stream<gpu>();
	auto s = mshadow::Stream<mxnet::gpu>::GetStream(stream);
	dim3 grid_dim = dim3(2, 8, 1);
	cuda::gatherpointKernel<<<grid_dim, 512, 0, s>>>(
		B, N, M, inputs[0].dptr<float>(), inputs[1].dptr<int>(), outputs[0].dptr<float>());
}

template<>
void GatherPointGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
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
	auto *stream = ctx.get_stream<gpu>();
	auto s = mshadow::Stream<mxnet::gpu>::GetStream(stream);
	dim3 grid_dim = dim3(2, 8, 1);
	cuda::scatteraddpointKernel<<<grid_dim, 512, 0, s>>>(
		B, N, M, out_grad.dptr<float>(), idx.dptr<int>(), in_grad.dptr<float>());
}

NNVM_REGISTER_OP(_contrib_GatherPoint)
.set_attr<FCompute>("FCompute<gpu>", GatherPointCompute<gpu>);

NNVM_REGISTER_OP(_backward_contrib_GatherPoint)
.set_attr<FCompute>("FCompute<gpu>", GatherPointGradCompute<gpu>);

} // namespace op
} // namespace mxnet
