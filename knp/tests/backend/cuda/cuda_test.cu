/**
 * @file cuda_test.cu
 * @brief CUDA backend test.
 * @kaspersky_support Artiom N.
 * @date 26.02.2025
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

#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_common.h>

#include <functional>
#include <iostream>
#include <vector>

#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/safe_call.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/vector.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_bus/message_bus.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_bus/messaging.cuh"
#include "../../../backends/gpu/cuda-backend/impl/uid.cuh"


namespace knp::testing
{

template <class Endpoint>
bool send_messages_smallest_network(const knp::core::UID &in_channel_uid, Endpoint &endpoint, knp::core::Step step)
{
    if (step % 5 == 0)
    {
        knp::core::messaging::SpikeMessage message{{in_channel_uid, 0}, {0}};
        endpoint.send_message(message);
        return true;
    }
    return false;
}


template <class Endpoint>
bool receive_messages_smallest_network(const knp::core::UID &out_channel_uid, Endpoint &endpoint)
{
    endpoint.receive_all_messages();
    // Write the steps on which the network sends a spike.
    if (!endpoint.template unload_messages<knp::core::messaging::SpikeMessage>(out_channel_uid).empty()) return true;
    return false;
}


__device__ void prn()
{
    printf("Hello from GPU! Thread ID: %d\n", threadIdx.x);

    knp::backends::gpu::cuda::device_lib::CUDAVector<int> cv;

    cv.reserve(2);

    printf("cv.size() = %lu\n", cv.size());
    cv.push_back(1);
    printf("cv.size() = %lu, v = %d\n", cv.size(), cv[0]);
    auto v = cv.pop_back();
    printf("p1 = %d\n", v);
    printf("cv.size() = %lu\n", cv.size());
    cv.resize(10);

    for (int i = 0; i < cv.size(); ++i) printf("i0 = %d\n", cv[i]);
    for (int i = 0; i < cv.size(); ++i) cv.set(i, i);
    for (int i = 0; i < cv.size(); ++i) printf("i1 = %d\n", cv[i]);
    cv.resize(5);
    for (int i = 0; i < cv.size(); ++i) printf("i2 = %d\n", cv[i]);
    cv.reserve(15);
    for (int i = 0; i < cv.size(); ++i) printf("i3 = %d\n", cv[i]);
}


__global__ void run_bus()
{
    prn();
}


/*TEST(CudaBackendSuite, CUDADevice)  // cppcheck-suppress syntaxError
{
    auto gpus = knp::devices::gpu::list_cuda_processors();
    for (const auto &gpu : gpus)
    {
        auto gpu_ptr = dynamic_cast<const knp::devices::gpu::CUDA *>(&gpu);
        SPDLOG_INFO(
            "GPU name: {}, warp size = {}, power = {}", gpu.get_name(), gpu_ptr->get_warp_size(), gpu.get_power());
    }
}*/


TEST(CudaBackendSuite, CudaUidConversionTest)
{
    knp::core::UID orig_uid;
    knp::backends::gpu::cuda::UID cuda_uid = knp::backends::gpu::cuda::to_gpu_uid(orig_uid);
    knp::core::UID restored_uid = knp::backends::gpu::cuda::to_cpu_uid(cuda_uid);
    ASSERT_EQ(orig_uid, restored_uid);
}




TEST(CudaBackendSuite, CudaBusSubscription)
{
    auto error = cudaGetLastError();
    cudaDeviceReset();
    run_bus<<<1, 2>>>();
    call_and_check(cudaDeviceSynchronize());
    error = cudaGetLastError();
    ASSERT_EQ(error, cudaSuccess);
}


TEST(CudaBackendSuite, NeuronsGettingTest)
{
    // const knp::testing::MTestingBack backend;

    // auto s_neurons = backend.get_supported_neurons();

    // ASSERT_LE(s_neurons.size(), boost::mp11::mp_size<knp::neuron_traits::AllNeurons>());
    // ASSERT_EQ(s_neurons[0], "BLIFATNeuron");
}


TEST(CudaBackendSuite, SynapsesGettingTest)
{
    // const knp::testing::MTestingBack backend;

    // auto s_synapses = backend.get_supported_synapses();

    // ASSERT_LE(s_synapses.size(), boost::mp11::mp_size<knp::synapse_traits::AllSynapses>());
    // ASSERT_EQ(s_synapses[0], "DeltaSynapse");
}

}  // namespace knp::testing
