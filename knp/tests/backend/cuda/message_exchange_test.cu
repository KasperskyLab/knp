/**
 * @file message_exchange_test.cu
 * @brief Message exchange via CUDA bus testing.
 * @kaspersky_support Artiom N.
 * @date 21.08.2025.
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

#include <tests_common.h>

#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/extraction.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/safe_call.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/vector.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_bus/message_bus.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_bus/messaging.cuh"
#include "../../../backends/gpu/cuda-backend/impl/uid.cuh"


namespace knp::testing
{
/*

TEST(CUDAMessagingSuite, SynapticImpactMessageSend)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    using SynapticImpactMessage = knp_cuda::SynapticImpactMessage;
    std::shared_ptr<knp::core::MessageBus> bus = knp::core::MessageBus::construct_zmq_bus();

    auto ep1{bus->create_endpoint()};
    knp::synapse_traits::OutputType synapse_type = knp::synapse_traits::OutputType::EXCITATORY;
    SynapticImpactMessage msg{
        {knp_cuda::UID{}},
        knp_cuda::UID{},
        knp_cuda::UID{},
        true,
        {{1, 2, synapse_type, 3, 4}, {4, 3, synapse_type, 2, 1}, {7, 8, synapse_type, 9, 10}}};

    auto &subscription = ep1.subscribe<SynapticImpactMessage>(knp_cuda::UID(), {msg.header_.sender_uid_});

    ep1.send_message(msg);
    // Message ID and message data.
    EXPECT_EQ(bus->route_messages(), 2);
    ep1.receive_all_messages();

    const auto &msgs = subscription.get_messages();

    EXPECT_EQ(msgs.size(), 1);
    EXPECT_EQ(msgs[0].header_.sender_uid_, msg.header_.sender_uid_);
    ASSERT_EQ(msgs[0].presynaptic_population_uid_, msg.presynaptic_population_uid_);
    ASSERT_EQ(msgs[0].postsynaptic_population_uid_, msg.postsynaptic_population_uid_);
    ASSERT_EQ(msgs[0].is_forcing_, msg.is_forcing_);
    ASSERT_EQ(msgs[0].impacts_, msg.impacts_);
}*/

} // namespace knp::testing
