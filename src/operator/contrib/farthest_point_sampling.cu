#include "./farthest_point_sampling-inl.h"
#include <mshadow/tensor.h>

namespace mxnet {
namespace cuda {

    __global__ void farthestpointsamplingKernel(int b,int n,int m
        ,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
      if (m<=0)
        return;
      const int BlockSize=512;
      __shared__ float dists[BlockSize];
      __shared__ int dists_i[BlockSize];
      const int BufferSize=3072;
      __shared__ float buf[BufferSize*3];
      for (int i=blockIdx.x;i<b;i+=gridDim.x){
        int old=0;
        if (threadIdx.x==0)
          idxs[i*m+0]=old;
        for (int j=threadIdx.x;j<n;j+=blockDim.x){
          temp[blockIdx.x*n+j]=1e38;
        }
        for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
          buf[j]=dataset[i*n*3+j];
        }
        __syncthreads();
        for (int j=1;j<m;j++){
          int besti=0;
          float best=-1;
          float x1=dataset[i*n*3+old*3+0];
          float y1=dataset[i*n*3+old*3+1];
          float z1=dataset[i*n*3+old*3+2];
          for (int k=threadIdx.x;k<n;k+=blockDim.x){
            float td=temp[blockIdx.x*n+k];
            float x2,y2,z2;
            if (k<BufferSize){
              x2=buf[k*3+0];
              y2=buf[k*3+1];
              z2=buf[k*3+2];
            }else{
              x2=dataset[i*n*3+k*3+0];
              y2=dataset[i*n*3+k*3+1];
              z2=dataset[i*n*3+k*3+2];
            }
            float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            float d2=min(d,td);
            if (d2!=td)
              temp[blockIdx.x*n+k]=d2;
            if (d2>best){
              best=d2;
              besti=k;
            }
          }
          dists[threadIdx.x]=best;
          dists_i[threadIdx.x]=besti;
          for (int u=0;(1<<u)<blockDim.x;u++){
            __syncthreads();
            if (threadIdx.x<(blockDim.x>>(u+1))){
              int i1=(threadIdx.x*2)<<u;
              int i2=(threadIdx.x*2+1)<<u;
              if (dists[i1]<dists[i2]){
                dists[i1]=dists[i2];
                dists_i[i1]=dists_i[i2];
              }
            }
          }
          __syncthreads();
          old=dists_i[0];
          if (threadIdx.x==0)
            idxs[i*m+j]=old;
        }
      }
    }

} // namespace cuda

namespace op {

template<>
void FarthestPointSamplingCompute<gpu>(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
						const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
using namespace mshadow;
CHECK_GT(ctx.requested.size(), 0);
	const int B = inputs[0].size(0);
	const int N = inputs[0].size(1);
    const FarthestPointSamplingParam& param = nnvm::get<FarthestPointSamplingParam>(attrs.parsed);

	auto *stream = ctx.get_stream<gpu>();
	auto s = mshadow::Stream<mxnet::gpu>::GetStream(stream);
    auto tmp_shape = mshadow::Shape2(32, N);

    Tensor<gpu, 2, float> tmp = ctx.requested[0].get_space_typed<gpu, 2, float>(tmp_shape, stream);
    // Fill<false>(stream, TBlob(reinterpret_cast<nnvm::dim_t*>(tmp.dptr_), tmp_shape, gpu::kDevMask), kWriteTo, 1e 10);

	cuda::farthestpointsamplingKernel<<<32, 512, 0, s>>>(
		B, N, param.npoints, inputs[0].dptr<float>(), tmp.dptr_, outputs[0].dptr<int>());
}

template<>
void FarthestPointSamplingGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx, 
							const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {

}

NNVM_REGISTER_OP(_contrib_FarthestPointSampling)
.set_attr<FCompute>("FCompute<gpu>", FarthestPointSamplingCompute<gpu>);

NNVM_REGISTER_OP(_backward_contrib_FarthestPointSampling)
.set_attr<FCompute>("FCompute<gpu>", FarthestPointSamplingGradCompute<gpu>);

} // namespace op
} // namespace mxnet
