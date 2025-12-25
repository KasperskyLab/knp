/**
 * @file output_test.cpp
 * @brief Tests for output channels and converters.
 * @kaspersky_support A. Vartenkov
 * @date 01.06.2023
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <knp/core/message_bus.h>
#include <knp/framework/io/out_converters/convert_bitwise.h>
#include <knp/framework/io/out_converters/convert_count.h>
#include <knp/framework/io/out_converters/convert_set.h>
#include <knp/framework/io/output_channel.h>

#include <tests_common.h>

#include <set>
#include <vector>


TEST(OutputSuite, ConverterTest)
{
    knp::core::UID sender_uid;
    // All indexes over 7 should be ignored.
    knp::core::messaging::SpikeMessage msg_0{{sender_uid, 1}, {1, 3, 8}};
    knp::core::messaging::SpikeMessage msg_1{{sender_uid, 3}, {1, 4, 10}};
    knp::core::messaging::SpikeMessage msg_2{{sender_uid, 5}, {1, 4, 7, 12}};

    std::vector<knp::core::messaging::SpikeMessage> messages{msg_0, msg_1, msg_2};

    auto result_count = knp::framework::io::output::converter_count(messages, 8);
    decltype(result_count) expected_count{0, 3, 0, 1, 2, 0, 0, 1};
    ASSERT_EQ(result_count, expected_count);

    auto bitwise_result = knp::framework::io::output::converter_bitwise(messages, 8);
    decltype(bitwise_result) expected_bitwise{false, true, false, true, true, false, false, true};
    ASSERT_EQ(bitwise_result, expected_bitwise);

    auto set_result = knp::framework::io::output::ConvertToSet(8)(messages);
    decltype(set_result) expected_set{1, 3, 4, 7};
    ASSERT_EQ(set_result, expected_set);
}


TEST(OutputSuite, ChannelTest)
{
    std::shared_ptr<knp::core::MessageBus> bus = knp::core::MessageBus::construct_bus();
    auto endpoint = bus->create_endpoint();
    // New random UID.
    knp::core::UID sender_uid;

    const size_t out_size = 8;
    auto count_converter = [out_size](const std::vector<knp::core::messaging::SpikeMessage> &messages)
    { return knp::framework::io::output::converter_count(messages, out_size); };
    auto set_converter = knp::framework::io::output::ConvertToSet(out_size);
    auto max_converter = [&](const std::vector<knp::core::messaging::SpikeMessage> &messages) -> size_t
    {
        auto result_container = count_converter(messages);
        return std::max_element(result_container.begin(), result_container.end()) - result_container.begin();
    };

    // Initialize counting channel.
    auto endpoint_1 = bus->create_endpoint();
    auto c1_uid = knp::core::UID();
    endpoint_1.subscribe<knp::core::messaging::SpikeMessage>(c1_uid, {sender_uid});
    knp::framework::io::output::OutputChannel channel_count{c1_uid, std::move(endpoint_1)};

    // Initialize neuron set channel.
    auto endpoint_2 = bus->create_endpoint();
    auto c2_uid = knp::core::UID();
    endpoint_2.subscribe<knp::core::messaging::SpikeMessage>(c2_uid, {sender_uid});
    knp::framework::io::output::OutputChannel channel_set{c2_uid, std::move(endpoint_2)};

    // Create a custom "max activations by neuron" channel.
    auto endpoint_3 = bus->create_endpoint();
    auto c3_uid = knp::core::UID();
    endpoint_3.subscribe<knp::core::messaging::SpikeMessage>(c3_uid, {sender_uid});
    knp::framework::io::output::OutputChannel channel_max{c3_uid, std::move(endpoint_3)};

    // Do message exchange.

    // const knp::core::messaging::SpikeMessage msg_4{{sender_uid, 6}, {1, 2, 4, 7, 10}};
    // The message with delay 0 will be ignored.
    endpoint.send_message(knp::core::messaging::SpikeMessage{{sender_uid, 0}, {0, 1, 2, 3, 4, 5}});
    // All indexes over 7 should also be ignored.
    endpoint.send_message(knp::core::messaging::SpikeMessage{{sender_uid, 1}, {1, 3, 8}});
    endpoint.send_message(knp::core::messaging::SpikeMessage{{sender_uid, 3}, {1, 4, 10}});
    endpoint.send_message(knp::core::messaging::SpikeMessage{{sender_uid, 5}, {1, 4, 7, 12}});
    bus->route_messages();
    endpoint.receive_all_messages();

    // Use channels.
    std::vector<size_t> count_result =
        knp::framework::io::output::output_channel_get<std::vector<size_t>>(channel_count, count_converter, 1, 5);
    // Expected result. Each number corresponds to the target neuron count of each index in messages.
    decltype(count_result) expected_count{0, 3, 0, 1, 2, 0, 0, 1};

    std::set<knp::core::messaging::SpikeIndex> set_result =
        knp::framework::io::output::output_channel_get<std::set<knp::core::messaging::SpikeIndex>>(
            channel_set, set_converter, 1, 5);
    // Expected result.
    decltype(set_result) expected_set{1, 3, 4, 7};

    size_t index = knp::framework::io::output::output_channel_get<size_t>(channel_max, max_converter, 1, 5);
    // Expected result.
    decltype(index) expected_index = 1;

    // Compare with expected result.
    ASSERT_EQ(count_result, expected_count);
    ASSERT_EQ(set_result, expected_set);
    ASSERT_EQ(index, expected_index);
}
