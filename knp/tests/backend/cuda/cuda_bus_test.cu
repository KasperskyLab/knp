/**
 * @file cuda_bus_test.cu
 * @brief CUDA backend test.
 * @kaspersky_support Vartenkov An.
 * @date 08.09.2025
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

#include <knp/core/message_bus.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_common.h>

#include <functional>
#include <iostream>
#include <vector>

#include "../../../backends/gpu/cuda-backend/impl/backend_impl.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/extraction.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/register_type.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/safe_call.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/vector.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_bus/message_bus.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_bus/messaging.cuh"
#include "../../../backends/gpu/cuda-backend/impl/uid.cuh"
#include "../../../backends/gpu/cuda-backend/impl/projection.cuh"
#include "../../../backends/gpu/cuda-backend/impl/population.cuh"


REGISTER_CUDA_VECTOR_TYPE(uint64_t);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::PopulationVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::ProjectionVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SpikeMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpact);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpactMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>::NeuronParameters);


namespace knp::testing
{

struct MessageBusTandem
{
    MessageBusTandem() :
            cpu_(std::move(*knp::core::MessageBus::construct_bus())),
            endpoint_(cpu_.create_endpoint()),
            gpu_(endpoint_)
    {}
    knp::core::MessageBus cpu_;
    knp::core::MessageEndpoint endpoint_;
    knp::backends::gpu::cuda::CUDAMessageBus gpu_;
};


TEST(CudaBackendSuite, MessagesTest)
{
    namespace knp_cuda = knp::backends::gpu::cuda;

    knp_cuda::SpikeMessage message_1;
    knp_cuda::SynapticImpactMessage message_2;
    ASSERT_EQ(message_1.neuron_indexes_.size(), 0);
    ASSERT_EQ(message_2.impacts_.size(), 0);
}


TEST(CudaBackendSuite, CudaHostSubscription)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    call_and_check(cudaDeviceReset());
    knp_cuda::UID receiver_uid = knp_cuda::to_gpu_uid(knp::core::UID{});
    knp_cuda::UID sender_1 = knp_cuda::to_gpu_uid(knp::core::UID{}), sender_2 = knp_cuda::to_gpu_uid(knp::core::UID{});
    knp_cuda::UID sender_3 = knp_cuda::to_gpu_uid(knp::core::UID{}), sender_4 = knp_cuda::to_gpu_uid(knp::core::UID{});
    ASSERT_NE(sender_1, sender_2);
    auto type_index = boost::mp11::mp_find<knp_cuda::MessageVariant, knp_cuda::SpikeMessage>();
    std::cout << "Make subscription with index " << type_index << std::endl;
    knp_cuda::Subscription subscription(receiver_uid, std::vector{sender_1, sender_2, sender_3}, type_index);
    std::cout << "Done making" << std::endl;
    ASSERT_EQ(subscription.get_senders().size(), 3);
    ASSERT_TRUE(subscription.has_sender(sender_2));
    ASSERT_FALSE(subscription.has_sender(sender_4));
}


TEST(CudaBackendSuite, BusSubscriptionsTest)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    MessageBusTandem message_buses;
    knp_cuda::UID uid_1{knp_cuda::new_uid()}, uid_2{knp_cuda::new_uid()};
    knp_cuda::UID uid_3{knp_cuda::new_uid()}, uid_4{knp_cuda::new_uid()};

    std::vector<knp_cuda::UID> uid_vec1{uid_2, uid_3};

    message_buses.gpu_.subscribe_gpu<knp_cuda::SpikeMessage>(uid_1, uid_vec1);
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 1);
    std::cout << "Number of subscriptions is 1" << std::endl;

    std::vector<knp_cuda::UID> uid_vec2{uid_3, uid_4};
    message_buses.gpu_.subscribe_gpu<knp_cuda::SpikeMessage>(uid_2, uid_vec2);
    std::cout << "Added subscription" << std::endl;
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 2);
    std::cout << "Number of subscriptions is 2" << std::endl;

    std::vector<knp_cuda::UID> uid_vec3{uid_1};

    message_buses.gpu_.subscribe_gpu<knp_cuda::SpikeMessage>(uid_1, uid_vec3);
    // TODO check corresponding subscription size, should become 3.
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 2);
    std::cout << "Number of subscriptions is still 2" << std::endl;
    // unsubscribing from a wrong type of messages, shouldn't change anything.
    message_buses.gpu_.unsubscribe<knp_cuda::SynapticImpactMessage>(uid_1);
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 2);
    std::cout << "Shouldn't have been deleted" << std::endl;
    // now unsubscribing from the right type of messages.
    message_buses.gpu_.unsubscribe<knp_cuda::SpikeMessage>(uid_1);
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 1);
}


TEST(CUDAMessagingSuite, AddReceiveBusMessage)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    using SpikeMessage = knp_cuda::SpikeMessage;

    MessageBusTandem message_buses;
    SpikeMessage msg{{knp_cuda::new_uid()}, {1, 2, 3, 4, 5}};
    knp_cuda::UID receiver_uid = knp_cuda::new_uid();
    std::vector<knp_cuda::UID> senders{msg.header_.sender_uid_};
    message_buses.gpu_.subscribe_gpu<SpikeMessage>(receiver_uid, {msg.header_.sender_uid_});
    EXPECT_EQ(message_buses.gpu_.unload_messages<SpikeMessage>(receiver_uid).size(), 0);
    message_buses.gpu_.send_message(msg);
    EXPECT_EQ(message_buses.gpu_.unload_messages<SpikeMessage>(receiver_uid).size(), 1);
    EXPECT_EQ(message_buses.gpu_.unload_messages<knp_cuda::SynapticImpactMessage>(receiver_uid).size(), 0);
    EXPECT_EQ(message_buses.gpu_.unload_messages<SpikeMessage>(msg.header_.sender_uid_).size(), 0);
    message_buses.gpu_.clear();
    EXPECT_EQ(message_buses.gpu_.unload_messages<SpikeMessage>(receiver_uid).size(), 0);
}

} // namespace knp::testing
