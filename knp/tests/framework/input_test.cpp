/**
 * @file input_test.cpp
 * @brief Input channel and converter tests.
 * @kaspersky_support A. Vartenkov
 * @date 31.06.2023
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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
#include <knp/core/messaging/messaging.h>
#include <knp/framework/io/in_converters/index_converter.h>
#include <knp/framework/io/in_converters/sequence_converter.h>
#include <knp/framework/io/input_channel.h>
#include <knp/framework/io/input_interpreters.h>

#include <tests_common.h>


TEST(InputSuite, SequenceConverterTest)
{
    auto stream = std::make_unique<std::stringstream>();
    // Float values equivalent to (0 1 1 0 0 1 1 0 1 0).
    *stream << "0.7 1.1 1.0 -0.2 0.1 3 2 0.7 11 -1";
    knp::framework::io::input::SequenceConverter<float> converter(
        std::move(stream), knp::framework::io::input::interpret_with_threshold<float>(1.0f), 10);

    auto result = converter();
    knp::core::messaging::SpikeData expected{1, 2, 5, 6, 8};

    ASSERT_EQ(result, expected);
}


TEST(InputSuite, IndexConverterTest)
{
    auto stream = std::make_unique<std::stringstream>();
    *stream << "1,3, 5 \n1  ,2 , 5\n3,5";
    knp::framework::io::input::IndexConverter converter(std::move(stream), ',');
    auto result = converter();
    knp::core::messaging::SpikeData expected_result{1, 3, 5};
    ASSERT_EQ(result, expected_result);
    result = converter();
    expected_result = {1, 2, 5};
    ASSERT_EQ(result, expected_result);
    result = converter();
    expected_result = {3, 5};
    ASSERT_EQ(result, expected_result);
}


TEST(InputSuite, ChannelTest)
{
    std::shared_ptr<knp::core::MessageBus> bus = knp::core::MessageBus::construct_bus();
    auto endpoint = bus->create_endpoint();

    auto converter = knp::framework::io::input::SequenceConverter<int>{
        std::make_unique<std::stringstream>(), knp::framework::io::input::interpret_as_bool<int>, 10};
    knp::framework::io::input::InputChannel channel{
        knp::core::UID(), bus->create_endpoint(), [&converter](auto size) { return converter(); }};

    auto &stream = dynamic_cast<std::stringstream &>(converter.get_stream());

    // Connect to output.
    knp::core::UID output_uid;
    knp::framework::io::input::connect_input(channel, endpoint, output_uid);

    // Send 12 integers to stream. Test that the final ones do not get into the message.
    stream << "1 0 1 1 0 1 1 1 1 0 1 1";
    knp::core::messaging::SpikeData expected_indexes = {0, 2, 3, 5, 6, 7, 8};
    const knp::core::Step send_time = 77;

    // Make channel send a message.
    channel.send(send_time);

    // Process messages.
    bus->route_messages();
    endpoint.receive_all_messages();

    auto messages = endpoint.unload_messages<knp::core::messaging::SpikeMessage>(output_uid);
    ASSERT_EQ(messages.size(), 1);

    auto message = messages[0];
    ASSERT_EQ(message.header_.sender_uid_, channel.get_uid());
    ASSERT_EQ(message.header_.send_time_, send_time);
    ASSERT_EQ(message.neuron_indexes_, expected_indexes);
}
