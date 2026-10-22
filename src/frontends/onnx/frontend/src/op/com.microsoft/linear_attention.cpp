// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "utils/attention.hpp"
#include "utils/common.hpp"

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector linear_attention(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 6, 6);
    const auto inputs = node.get_ov_inputs();
    const auto heads = node.get_attribute_value<int64_t>("q_num_heads");
    const auto kv_heads = node.get_attribute_value<int64_t>("kv_num_heads");
    FRONT_END_OP_CONVERSION_CHECK(node.get_attribute_value<std::string>("update_rule", "gated_delta") == "gated_delta",
                                  "LinearAttention: only update_rule='gated_delta' is supported.");
    FRONT_END_OP_CONVERSION_CHECK(heads > 0 && heads == kv_heads,
                                  "LinearAttention: positive, equal q_num_heads and kv_num_heads are required.");
    FRONT_END_OP_CONVERSION_CHECK(node.get_attribute_value<int64_t>("state_window", 0) == 0,
                                  "LinearAttention: state_window must be 0.");

    for (const auto index : {0, 1, 2, 4, 5}) {
        FRONT_END_OP_CONVERSION_CHECK(common::is_input_valid(inputs, index),
                                      "LinearAttention: query, key, value, decay and beta are required.");
        FRONT_END_OP_CONVERSION_CHECK(inputs[index].get_element_type() == inputs[0].get_element_type(),
                                      "LinearAttention: packed inputs must have the same element type.");
        const auto& shape = inputs[index].get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(shape.rank().compatible(3), "LinearAttention: packed inputs must have rank 3.");
        if (shape.rank().is_static()) {
            if (shape[2].is_static()) {
                const auto width = shape[2].get_length();
                FRONT_END_OP_CONVERSION_CHECK(
                    index < 3 ? (width > 0 && width % heads == 0) : (width == heads || (index == 5 && width == 1)),
                    "LinearAttention: invalid packed width; decay must be per-head and beta per-head or shared.");
            }
            const auto& query_shape = inputs[0].get_partial_shape();
            if (query_shape.rank().is_static()) {
                FRONT_END_OP_CONVERSION_CHECK(
                    shape[0].compatible(query_shape[0]) && shape[1].compatible(query_shape[1]),
                    "LinearAttention: batch and sequence dimensions must match.");
            }
        }
    }

    using namespace ov::op;
    const auto head_count = v0::Constant::create(element::i64, Shape{1}, {heads});
    const auto unpack_pattern = v0::Constant::create<int64_t>(element::i64, Shape{4}, {0, 0, heads, -1});
    Output<ov::Node> query = std::make_shared<v1::Reshape>(inputs[0], unpack_pattern, true);
    const auto key = std::make_shared<v1::Reshape>(inputs[1], unpack_pattern, true);
    const auto value = std::make_shared<v1::Reshape>(inputs[2], unpack_pattern, true);
    const auto query_shape = std::make_shared<v3::ShapeOf>(query);
    const auto value_shape = std::make_shared<v3::ShapeOf>(value);
    const auto key_width = attention::get_dimensions(query_shape, {3});

    const auto scale = node.get_attribute_value<float>("scale", 0.0f);
    FRONT_END_OP_CONVERSION_CHECK(std::isfinite(scale), "LinearAttention: scale must be finite.");
    if (scale != 0.0f) {
        // GatedDeltaNet always applies 1/sqrt(d_k); compensate for an explicit ONNX scale.
        const auto width_f32 = std::make_shared<v0::Convert>(key_width, element::f32);
        const auto factor = std::make_shared<v1::Multiply>(std::make_shared<v0::Sqrt>(width_f32),
                                                           v0::Constant::create(element::f32, Shape{}, {scale}));
        query = std::make_shared<v1::Multiply>(query, std::make_shared<v1::ConvertLike>(factor, query));
    }

    Output<ov::Node> state;
    if (common::is_input_valid(inputs, 3)) {
        state = inputs[3];
        const auto& state_shape = state.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(state_shape.rank().compatible(4),
                                      "LinearAttention: past_state must have rank 4.");
        if (state_shape.rank().is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(state_shape[0].compatible(query.get_partial_shape()[0]),
                                          "LinearAttention: past_state batch must match query.");
        }
    } else {
        const auto state_shape = std::make_shared<v0::Concat>(OutputVector{attention::get_dimensions(query_shape, {0}),
                                                                           head_count,
                                                                           key_width,
                                                                           attention::get_dimensions(value_shape, {3})},
                                                              0);
        state = std::make_shared<v3::Broadcast>(v0::Constant::create(inputs[0].get_element_type(), Shape{}, {0}),
                                                state_shape);
    }

    const auto gate_shape =
        std::make_shared<v0::Concat>(OutputVector{attention::get_dimensions(query_shape, {0, 1}), head_count}, 0);
    const auto beta = std::make_shared<v3::Broadcast>(inputs[5], gate_shape);
    const auto gdn = std::make_shared<internal::GatedDeltaNet>(query, key, value, state, inputs[4], beta, false);
    const auto output = std::make_shared<v1::Reshape>(gdn->output(0), std::make_shared<v3::ShapeOf>(inputs[2]), false);
    return {output, gdn->output(1)};
}

ONNX_OP("LinearAttention", OPSET_SINCE(1), com_microsoft::opset_1::linear_attention, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
