/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file my_predict_api.cc
 * \brief C predict API of mxnet
 */
#include <dmlc/base.h>
#include <dmlc/memory_io.h>
#include <mxnet/my_predict_api.h>
#include <mxnet/executor.h>
#include <mxnet/ndarray.h>
#include <nnvm/pass_functions.h>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "./c_api_common.h"
#include "../operator/operator_common.h"
#include "../executor/exec_pass.h"

using namespace mxnet;
using nnvm::Symbol;

//temp
struct MXATMP
{
	Symbol sym;

	std::vector<std::string> arg_names;
	std::vector<std::string> aux_names;
	std::vector<TShape> arg_shapes;
	std::vector<TShape> aux_shapes;
	std::vector<NDArray> aux_arrays;
	//std::unordered_map<std::string, NDArray> arg_params;
	//std::unordered_map<std::string, NDArray> aux_params;

	mx_uint num_input_nodes;
	char input_keys[99][256];
	Context ctx;
	mx_uint input_shape_indptr[100];
};

// predictor interface
struct MXAPIPredictor 
{
  // output arrays
  std::vector<NDArray> out_arrays;
  // argument arrays
  std::vector<NDArray> arg_arrays;
  // output shapes
  std::vector<TShape> out_shapes;
  // uint32_t buffer for output shapes
  std::vector<uint32_t> out_shapes_buffer;
  // key to arguments
  std::unordered_map<std::string, size_t> key2arg;
  // executor
  std::unique_ptr<Executor> exec;

  MXATMP *tmp;
};


struct MXAPINDList
{
  std::vector<std::string> keys;
  std::vector<TShape> shapes;
  std::vector<uint32_t> shapes_buffer;
  std::vector<size_t> indptr;
  std::vector<mx_float> data;
};

int MYPredCreate(const char* symbol_json_str,
                 const void* param_bytes,
                 int param_size,
                 int dev_type, int dev_id,
                 mx_uint num_input_nodes,
                 const char** input_keys,
                 const mx_uint* input_shape_indptr,
                 const mx_uint* input_shape_data,
                 PredictorHandle* out)
{
  return MYPredCreatePartialOut(
      symbol_json_str,
      param_bytes,
      param_size,
      dev_type,
      dev_id,
      num_input_nodes,
      input_keys,
      input_shape_indptr,
      input_shape_data,
      0,
      NULL,
      out);
}
namespace mxnet {

}  // namespace mxnet

int MYPredCreatePartialOut(const char* symbol_json_str,
                           const void* param_bytes,
                           int param_size,
                           int dev_type, int dev_id,
                           mx_uint num_input_nodes,
                           const char** input_keys,
                           const mx_uint* input_shape_indptr,
                           const mx_uint* input_shape_data,
                           mx_uint num_output_nodes,
                           const char** output_keys,
                           PredictorHandle* out) 
{
  using nnvm::Symbol;
  MXAPIPredictor* ret = new MXAPIPredictor();

  ret->tmp = new MXATMP();
  
  API_BEGIN();
  Symbol sym;
  // make sure symbols are registered
  {
  mx_uint outSize;
  const char **outArray;
  MXListAllOpNames(&outSize, &outArray);
  }
  // load in the symbol.
  {
    nnvm::Graph g;
    g.attrs["json"] = std::make_shared<nnvm::any>(std::string(symbol_json_str));
    sym.outputs = nnvm::ApplyPass(g, "LoadLegacyJSON").outputs;
  }
  // looks likely to output the internal results
  if (num_output_nodes != 0) 
  {
    Symbol internal = sym.GetInternals();
    std::vector<std::string> all_out = internal.ListOutputNames();
    std::vector<Symbol> out_syms(num_output_nodes);
    for (mx_uint i = 0; i < num_output_nodes; ++i) {
      std::string out_key(output_keys[i]);
      out_key += "_output";
      for (size_t j = 0; j < all_out.size(); ++j) {
        if (all_out[j] == out_key) {
          out_syms[i] = internal[j];
          break;
        }
        CHECK_NE(j, all_out.size() - 1) << "didn't find node name: " << out_key;
      }
    }
    sym = nnvm::Symbol::CreateGroup(out_syms);
  }

  // load the parameters
  std::unordered_map<std::string, NDArray> arg_params, aux_params;
  {
    std::unordered_set<std::string> arg_names, aux_names;
    std::vector<std::string> arg_names_vec = sym.ListInputNames(Symbol::kReadOnlyArgs);
    std::vector<std::string> aux_names_vec = sym.ListInputNames(Symbol::kAuxiliaryStates);
    for (size_t i = 0; i < arg_names_vec.size(); ++i) 
    {
      arg_names.insert(arg_names_vec[i]);
    }
    for (size_t i = 0; i < aux_names_vec.size(); ++i) 
    {
      aux_names.insert(aux_names_vec[i]);
    }
    std::vector<NDArray> data;
    std::vector<std::string> names;
    dmlc::MemoryFixedSizeStream fi((void*)param_bytes, param_size);  // NOLINT(*)
    NDArray::Load(&fi, &data, &names);
    CHECK_EQ(names.size(), data.size())
        << "Invalid param file format";
    for (size_t i = 0; i < names.size(); ++i) 
    {
      if (!strncmp(names[i].c_str(), "aux:", 4)) 
      {
        std::string name(names[i].c_str() + 4);
        if (aux_names.count(name) != 0) 
        {
          aux_params[name] = data[i];
        }
      }
      if (!strncmp(names[i].c_str(), "arg:", 4)) 
      {
        std::string name(names[i].c_str() + 4);
        if (arg_names.count(name) != 0) 
        {
          arg_params[name] = data[i];
        }
      }
    }
  }
  // shape inference and bind
  std::unordered_map<std::string, TShape> known_shape;
  for (mx_uint i = 0; i < num_input_nodes; ++i) 
  {
    known_shape[std::string(input_keys[i])] =
        TShape(input_shape_data + input_shape_indptr[i],
               input_shape_data + input_shape_indptr[i + 1]);
  }
  std::vector<std::string> arg_names = sym.ListInputNames(Symbol::kReadOnlyArgs);
  std::vector<std::string> aux_names = sym.ListInputNames(Symbol::kAuxiliaryStates);
  std::vector<TShape> out_shapes(sym.ListOutputNames().size());
  std::vector<TShape> aux_shapes(aux_names.size());
  std::vector<TShape> arg_shapes;

  for (size_t i = 0; i < arg_names.size(); ++i) 
  {
    std::string key = arg_names[i];
    ret->key2arg[key] = i;
  }

  try 
  {
    std::vector<TShape> in_shapes;
    for (std::string key : sym.ListInputNames(Symbol::kAll)) 
    {
      if (known_shape.count(key) != 0) 
      {
        in_shapes.push_back(known_shape[key]);
      } 
      else 
      {
        in_shapes.push_back(TShape());
      }
    }
    nnvm::Graph g;
    g.outputs = sym.outputs;
    g = mxnet::exec::InferShape(std::move(g), std::move(in_shapes), "__shape__");
    bool infer_complete = (g.GetAttr<size_t>("shape_num_unknown_nodes") == 0);
    CHECK(infer_complete)
      << "The shape information of is not enough to get the shapes";
    CopyAttr(g.indexed_graph(),
             g.GetAttr<nnvm::ShapeVector>("shape"),
             &arg_shapes, &out_shapes, &aux_shapes);
  }
   catch (const mxnet::op::InferShapeError &err) 
  {
    throw dmlc::Error(err.msg);
  }

  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  ret->tmp->ctx = ctx;
  
  std::vector<NDArray> arg_arrays, aux_arrays;
  for (size_t i = 0; i < arg_shapes.size(); ++i) 
  {
    NDArray nd = NDArray(arg_shapes[i], ctx);
    if (arg_params.count(arg_names[i]) != 0) 
    {
      CopyFromTo(arg_params[arg_names[i]], &nd);
    }
    arg_arrays.push_back(nd);
  }
  for (size_t i = 0; i < aux_shapes.size(); ++i) 
  {
    NDArray nd = NDArray(aux_shapes[i], ctx);
    if (aux_params.count(aux_names[i]) != 0) 
    {
      CopyFromTo(aux_params[aux_names[i]], &nd);
    }
    aux_arrays.push_back(nd);
  }

  ret->tmp->sym = sym;
  ret->tmp->num_input_nodes = num_input_nodes;
  for (size_t i = 0; i < num_input_nodes; ++i)
  {
    strcpy(ret->tmp->input_keys[i], input_keys[i]);
	ret->tmp->input_shape_indptr[i] = input_shape_indptr[i];
  }
  ret->tmp->input_shape_indptr[num_input_nodes] = input_shape_indptr[num_input_nodes];

  ret->tmp->arg_names = arg_names;
  ret->tmp->aux_names = aux_names;
  ret->tmp->arg_shapes = arg_shapes;
  ret->tmp->aux_shapes = aux_shapes;
  ret->tmp->aux_arrays = aux_arrays;
  //ret->tmp->arg_params = arg_params;
  //ret->tmp->aux_params = aux_params;
  
  ret->arg_arrays = arg_arrays;

  // bind
  {
    std::map<std::string, Context> ctx_map;
    std::vector<NDArray> grad_store(arg_arrays.size());
    std::vector<OpReqType> grad_req(arg_arrays.size(), kNullOp);

    ret->exec.reset(Executor::Bind(sym, ctx, ctx_map,
                                   arg_arrays,
                                   grad_store, grad_req,
                                   aux_arrays));
    ret->out_shapes = out_shapes;
    ret->out_arrays = ret->exec->outputs();

  }
  *out = ret;
  API_END_HANDLE_ERROR(delete ret->tmp; delete ret);
}

/*Reshape*/
int MYPredReshape( PredictorHandle handle, const mx_uint* input_shape_data)
{

  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);

  API_BEGIN();

  // num_nodes
  mx_uint num_input_nodes = p->tmp->num_input_nodes;
  Context ctx = p->tmp->ctx;
  
  std::unordered_map<std::string, TShape> known_shape;
  for (mx_uint i = 0; i < num_input_nodes; ++i) 
  {
    known_shape[std::string(p->tmp->input_keys[i])] =
        TShape(input_shape_data + p->tmp->input_shape_indptr[i],
               input_shape_data + p->tmp->input_shape_indptr[i + 1]);
  }
  std::vector<std::string> &arg_names = p->tmp->arg_names;
  std::vector<std::string> &aux_names = p->tmp->aux_names;
  std::vector<TShape> out_shapes(p->tmp->sym.ListOutputNames().size());
  std::vector<TShape> aux_shapes(aux_names.size());
  std::vector<TShape> arg_shapes;

  try 
  {
    std::vector<TShape> in_shapes;
    for (std::string key : p->tmp->sym.ListInputNames(Symbol::kAll))
    {
      if (known_shape.count(key) != 0) 
      {
        in_shapes.push_back(known_shape[key]);
      } 
      else 
      {
        in_shapes.push_back(TShape());
      }
    }
    nnvm::Graph g;
    g.outputs = p->tmp->sym.outputs;
    g = mxnet::exec::InferShape(std::move(g), std::move(in_shapes), "__shape__");
    bool infer_complete = (g.GetAttr<size_t>("shape_num_unknown_nodes") == 0);
    CHECK(infer_complete)
      << "The shape information of is not enough to get the shapes";
    CopyAttr(g.indexed_graph(),
             g.GetAttr<nnvm::ShapeVector>("shape"),
             &arg_shapes, &out_shapes, &aux_shapes);
  }
   catch (const mxnet::op::InferShapeError &err) 
  {
    throw dmlc::Error(err.msg);
  }
  
   for (size_t i = 0; i < arg_shapes.size(); ++i) 
  {
	  if (p->tmp->arg_shapes[i] != arg_shapes[i])
	  {
		NDArray nd = NDArray(arg_shapes[i], ctx);
		p->arg_arrays[i] = nd;
	  }
  }
  
  for (size_t i = 0; i < aux_shapes.size(); ++i) 
  {
	  if (p->tmp->aux_shapes[i] != aux_shapes[i])
	  {
		  NDArray nd = NDArray(aux_shapes[i], ctx);
		  p->tmp->aux_arrays[i] = nd;

	  }
  }
   //std::vector<NDArray> arg_arrays, aux_arrays;
   //for (size_t i = 0; i < arg_shapes.size(); ++i)
   //{
	  // NDArray nd = NDArray(arg_shapes[i], ctx);
	  // if (p->tmp->arg_params.count(arg_names[i]) != 0)
	  // {
		 //  CopyFromTo(p->tmp->arg_params[arg_names[i]], &nd);
	  // }
	  // arg_arrays.push_back(nd);
   //}
   //for (size_t i = 0; i < aux_shapes.size(); ++i)
   //{
	  // NDArray nd = NDArray(aux_shapes[i], ctx);
	  // if (p->tmp->aux_params.count(aux_names[i]) != 0)
	  // {
		 //  CopyFromTo(p->tmp->aux_params[aux_names[i]], &nd);
	  // }
	  // aux_arrays.push_back(nd);
   //}

   //p->arg_arrays = arg_arrays;

  // bind
  {
    std::map<std::string, Context> ctx_map;
    std::vector<NDArray> grad_store(p->arg_arrays.size());
    std::vector<OpReqType> grad_req(p->arg_arrays.size(), kNullOp);

    p->exec.reset(Executor::Bind(p->tmp->sym, ctx, ctx_map,
								   p->arg_arrays,
                                   grad_store, grad_req,
								   p->tmp->aux_arrays, p->exec.get()));
    p->out_shapes = out_shapes;
    p->out_arrays = p->exec->outputs();
  }
  API_END();
}

int MYPredGetOutputShape(PredictorHandle handle,
                         mx_uint out_index,
                         mx_uint** shape_data,
                         mx_uint* shape_ndim) 
{
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(out_index, p->out_arrays.size())
      << "Index exceed number of outputs";

  const TShape& s = p->out_shapes[out_index];
  p->out_shapes_buffer.resize(s.ndim());
  nnvm::ShapeTypeCast(s.begin(), s.end(), p->out_shapes_buffer.data());
  *shape_data = p->out_shapes_buffer.data();
  *shape_ndim = p->out_shapes[out_index].ndim();
  API_END();
}

int MYPredSetInput(PredictorHandle handle,
                   const char* key,
                   const mx_float* data,
                   mx_uint size) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  auto it = p->key2arg.find(key);
  if (it == p->key2arg.end()) {
    LOG(FATAL) << "cannot find input key " << key;
  }
  NDArray& nd = p->arg_arrays[it->second];
  nd.SyncCopyFromCPU(data, size);
  API_END();
}

int MYPredForward(PredictorHandle handle) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  p->exec->Forward(false);
  API_END();
}

int MYPredPartialForward(PredictorHandle handle, int step, int* step_left) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  p->exec->PartialForward(false, step, step_left);
  API_END();
}

int MYPredGetOutput(PredictorHandle handle,
                    mx_uint index,
                    mx_float* data,
                    mx_uint size) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->out_arrays.size())
      << "Output index out of range";
  const NDArray& nd = p->out_arrays[index];
  nd.SyncCopyToCPU(data, size);
  API_END();
}

int MYPredFree(PredictorHandle handle) {
  API_BEGIN();
  delete static_cast<MXAPIPredictor*>(handle)->tmp;
  delete static_cast<MXAPIPredictor*>(handle);
  API_END();
}

int MYNDListCreate(const char* nd_file_bytes,
                   int nd_file_size,
                   NDListHandle *out,
                   mx_uint* out_length) {
  MXAPINDList* ret = new MXAPINDList();
  API_BEGIN();
  std::vector<NDArray> arrays;
  dmlc::MemoryFixedSizeStream fi((void*)nd_file_bytes, nd_file_size);  // NOLINT(*)
  NDArray::Load(&fi,
                &(arrays),
                &(ret->keys));
  if (ret->keys.size() == 0) {
    ret->keys.resize(arrays.size());
  }
  ret->indptr.push_back(0);
  for (size_t i = 0; i < arrays.size(); ++i) {
    TShape shape = arrays[i].shape();
    size_t begin = ret->data.size();
    size_t size = shape.Size();
    ret->shapes.push_back(shape);
    ret->data.resize(begin + size);
    arrays[i].SyncCopyToCPU(dmlc::BeginPtr(ret->data) + begin, size);
    ret->indptr.push_back(begin + size);
  }
  *out = ret;
  *out_length = static_cast<mx_uint>(arrays.size());
  API_END();
}

int MYNDListGet(NDListHandle handle,
                mx_uint index,
                const char** out_key,
                const mx_float** out_data,
                const mx_uint** out_shape,
                mx_uint* out_ndim) {
  MXAPINDList* p = static_cast<MXAPINDList*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->shapes.size())
      << "Index out of range";
  *out_key = p->keys[index].c_str();
  *out_data = dmlc::BeginPtr(p->data) + p->indptr[index];
  const TShape& s = p->shapes[index];
  p->shapes_buffer.resize(s.ndim());
  nnvm::ShapeTypeCast(s.begin(), s.end(), p->shapes_buffer.data());
  *out_shape = p->shapes_buffer.data();
  *out_ndim = p->shapes[index].ndim();
  API_END();
}

int MYNDListFree(NDListHandle handle) {
  API_BEGIN();
  delete static_cast<MXAPINDList*>(handle);
  API_END();
}
