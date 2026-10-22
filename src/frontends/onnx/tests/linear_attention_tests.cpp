// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

#include <cmath>
#include <limits>
#include <sstream>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "onnx_utils.hpp"
#include "openvino/op/gated_delta_net.hpp"

namespace {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::ModelProto;

ModelProto make_model(bool with_state = true) {
    ModelProto model;
    model.set_ir_version(9);
    model.add_opset_import()->set_version(21);
    auto* opset = model.add_opset_import();
    opset->set_domain("com.microsoft");
    opset->set_version(1);
    auto* graph = model.mutable_graph();
    graph->set_name("linear_attention_gdn");
    auto* node = graph->add_node();
    node->set_op_type("LinearAttention");
    node->set_domain("com.microsoft");
    auto add_info = [](auto* info, const std::string& name, const std::vector<int64_t>& shape) {
        info->set_name(name);
        auto* type = info->mutable_type()->mutable_tensor_type();
        type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        for (const auto dim : shape) {
            type->mutable_shape()->add_dim()->set_dim_value(dim);
        }
    };
    const std::vector<std::pair<std::string, std::vector<int64_t>>> inputs = {
        {"query", {1, 2, 4}},
        {"key", {1, 2, 4}},
        {"value", {1, 2, 6}},
        {"past_state", {1, 2, 2, 3}},
        {"decay", {1, 2, 2}},
        {"beta", {1, 2, 1}},
    };
    for (const auto& [name, shape] : inputs) {
        if (name == "past_state" && !with_state) {
            node->add_input("");
        } else {
            node->add_input(name);
            add_info(graph->add_input(), name, shape);
        }
    }
    for (const auto* name : {"q_num_heads", "kv_num_heads"}) {
        auto* attr = node->add_attribute();
        attr->set_name(name);
        attr->set_type(AttributeProto::INT);
        attr->set_i(2);
    }
    node->add_output("output");
    node->add_output("present_state");
    add_info(graph->add_output(), "output", {1, 2, 6});
    add_info(graph->add_output(), "present_state", {1, 2, 2, 3});
    return model;
}

std::shared_ptr<ov::Model> convert(const ModelProto& proto) {
    auto frontend = ov::frontend::FrontEndManager().load_by_framework("onnx");
    std::istringstream stream(proto.SerializeAsString());
    return frontend->convert(frontend->load(static_cast<std::istream*>(&stream)));
}

void set_int(ModelProto& model, const std::string& name, int64_t value) {
    auto* node = model.mutable_graph()->mutable_node(0);
    for (auto& attr : *node->mutable_attribute()) {
        if (attr.name() == name) {
            attr.set_i(value);
            return;
        }
    }
    auto* attr = node->add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::INT);
    attr->set_i(value);
}
}  // namespace

TEST(ONNXFeLinearAttention, converts_to_gdn_without_normalizing_inputs_again) {
    const auto model = convert(make_model());
    EXPECT_EQ(count_ops_of_type<ov::op::internal::GatedDeltaNet>(model), 1);
    for (const auto& op : model->get_ops()) {
        if (const auto gdn = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(op)) {
            EXPECT_FALSE(gdn->get_fuse_qk_l2norm());
        }
    }
    EXPECT_EQ(model->output(0).get_shape(), (ov::Shape{1, 2, 6}));
    EXPECT_EQ(model->output(1).get_shape(), (ov::Shape{1, 2, 2, 3}));
}

TEST(ONNXFeLinearAttention, preserves_recurrence_state_and_explicit_scale) {
    for (const auto with_state : {false, true}) {
        for (const auto scale : {0.0f, 0.25f, -0.5f}) {
            SCOPED_TRACE(testing::Message() << "state=" << with_state << ", scale=" << scale);
            auto proto = make_model(with_state);
            auto* attr = proto.mutable_graph()->mutable_node(0)->add_attribute();
            attr->set_name("scale");
            attr->set_type(AttributeProto::FLOAT);
            attr->set_f(scale);
            auto test = ov::test::TestCase(convert(proto), ov::test::utils::DEVICE_TEMPLATE);
            test.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8});
            test.add_input<float>({1, 0, 0, 1, 0, 1, 1, 0});
            test.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
            if (with_state) {
                test.add_input<float>(std::vector<float>(12, 1.0f));
            }
            test.add_input<float>({std::log(0.5f), std::log(0.25f), std::log(0.5f), std::log(0.25f)});
            test.add_input<float>({0.5f, 1.0f});
            std::vector<float> expected =
                with_state ? std::vector<float>{1.75f,
                                                2.25f,
                                                2.75f,
                                                9.25f,
                                                11.25f,
                                                13.25f,
                                                43.875f,
                                                51.125f,
                                                58.375f,
                                                74.25f,
                                                82.25f,
                                                90.25f}
                           : std::vector<float>{0.5f, 1, 1.5f, 8, 10, 12, 43.25f, 50.5f, 57.75f, 74, 82, 90};
            for (auto& value : expected) {
                value *= scale == 0.0f ? 1.0f / std::sqrt(2.0f) : scale;
            }
            test.add_expected_output<float>(expected);
            test.add_expected_output<float>(
                with_state
                    ? std::vector<float>{0.375f, 0.625f, 0.875f, 7, 8, 9, 10, 11, 12, 0.53125f, 0.65625f, 0.78125f}
                    : std::vector<float>{0.25f, 0.5f, 0.75f, 7, 8, 9, 10, 11, 12, 0.5f, 0.625f, 0.75f});
            test.run_with_tolerance_as_fp(2e-5f);
        }
    }
}

TEST(ONNXFeLinearAttention, rejects_unsupported_recurrences) {
    for (const auto* rule : {"linear", "gated", "delta", "invalid"}) {
        auto proto = make_model();
        auto* attr = proto.mutable_graph()->mutable_node(0)->add_attribute();
        attr->set_name("update_rule");
        attr->set_type(AttributeProto::STRING);
        attr->set_s(rule);
        EXPECT_THROW(convert(proto), ov::Exception);
    }
}

TEST(ONNXFeLinearAttention, rejects_unsupported_state_windows_and_head_counts) {
    for (const auto& [name, value] :
         std::vector<std::pair<std::string, int64_t>>{{"state_window", 1}, {"q_num_heads", 0}, {"kv_num_heads", 1}}) {
        auto proto = make_model();
        set_int(proto, name, value);
        EXPECT_THROW(convert(proto), ov::Exception);
    }
}

TEST(ONNXFeLinearAttention, rejects_per_key_decay) {
    auto proto = make_model();
    proto.mutable_graph()
        ->mutable_input(4)
        ->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(2)
        ->set_dim_value(4);
    EXPECT_THROW(convert(proto), ov::Exception);
}

TEST(ONNXFeLinearAttention, rejects_nonfinite_scale) {
    auto proto = make_model();
    auto* attr = proto.mutable_graph()->mutable_node(0)->add_attribute();
    attr->set_name("scale");
    attr->set_type(AttributeProto::FLOAT);
    attr->set_f(std::numeric_limits<float>::infinity());
    EXPECT_THROW(convert(proto), ov::Exception);
}

TEST(ONNXFeLinearAttention, defaults_missing_state_to_query_precision) {
    auto proto = make_model(false);
    for (auto& input : *proto.mutable_graph()->mutable_input()) {
        input.mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    }
    for (auto& output : *proto.mutable_graph()->mutable_output()) {
        output.mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    }
    const auto model = convert(proto);
    EXPECT_EQ(model->output(0).get_element_type(), ov::element::f16);
    EXPECT_EQ(model->output(1).get_element_type(), ov::element::f16);
}

TEST(ONNXFeLinearAttention, rejects_mismatched_state_batch) {
    auto proto = make_model();
    proto.mutable_graph()
        ->mutable_input(3)
        ->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(0)
        ->set_dim_value(2);
    EXPECT_THROW(convert(proto), ov::Exception);
}
