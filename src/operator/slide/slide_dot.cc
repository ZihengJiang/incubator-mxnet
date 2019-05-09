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
 * \file dot.cc
 * \brief CPU Implementation of matrix dot
 */

#include "./slide_dot-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(SlideDotParam);

void SlideDotForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const SlideDotParam& param = nnvm::get<SlideDotParam>(attrs.parsed);
  CHECK_EQ(outputs[0].type_flag_, kFloat32)
      << "only support float32 for now";
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";

  float *lhs = inputs[0].dptr<float>();
  float *rhs = inputs[1].dptr<float>();
  float *out = outputs[0].dptr<float>();

  // (m, l) x (l, n) -> (m, n)
  TShape ishape = inputs[0].shape_;
  TShape dshape = inputs[1].shape_;
  TShape oshape = outputs[0].shape_;
  dim_t m = oshape[0], n = oshape[1], l = ishape[1];

  for (dim_t i = 0; i < m; ++i) {
    for (dim_t j = 0; j < n; ++j) {
      float val = 0;
      for (dim_t k = 0; k < l; ++k) {
        val += lhs[i * l + k] * rhs[k * n + j];
      }
      out[i * n + j] = val;
    }
  }
}



NNVM_REGISTER_OP(slide_dot)
MXNET_ADD_SPARSE_OP_ALIAS(slide_dot)
.describe(R"doc(Dot product of two arrays.
)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SlideDotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", SlideDotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", SlideDotForwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SlideDotForward)
// .set_attr<FComputeEx>("FComputeEx<cpu>", SlideDotForwardEx<cpu>)
// .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_dot"})
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.add_arguments(SlideDotParam::__FIELDS__());

// NNVM_REGISTER_OP(_backward_dot)
// .set_num_inputs(3)
// .set_num_outputs(2)
// .set_attr_parser(ParamParser<DotParam>)
// .set_attr<nnvm::TIsBackward>("TIsBackward", true)
// .set_attr<FInferStorageType>("FInferStorageType", DotBackwardInferStorageType)
// .set_attr<FResourceReqonst SlideDotParam& param = nnvm::get<SlideDotParam>(attrs.parsed);

}  // namespace op
}  // namespace mxnet
