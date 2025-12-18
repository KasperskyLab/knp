/**
 * @file synchronization_test.cpp
 * @brief Synchronization tests for different backends.
 * @kaspersky_support Vartenkov An.
 * @date 17.05.24.
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

#include <knp/backends/cpu-multi-threaded/backend.h>
#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/core/population.h>
#include <knp/framework/synchronization.h>

#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_common.h>

#include <vector>


namespace knp::testing
{
class STestingBack : public knp::backends::single_threaded_cpu::SingleThreadedCPUBackend
{
public:
    STestingBack() = default;
    void _init() override { knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::_init(); }
};


class MTestingBack : public knp::backends::multi_threaded_cpu::MultiThreadedCPUBackend
{
public:
    MTestingBack() = default;
    void _init() override { knp::backends::multi_threaded_cpu::MultiThreadedCPUBackend::_init(); }
};

}  // namespace knp::testing


TEST(SynchronizationSuite, SingleThreadCpuTest)
{
    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.
    knp::testing::STestingBack backend;
    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};
    auto loop_projection =
        knp::testing::DeltaProjection{population.get_uid(), population.get_uid(), knp::testing::synapse_generator, 1};
    auto input_projection = knp::testing::DeltaProjection{
        knp::core::UID{false}, population.get_uid(), knp::testing::input_projection_gen, 1};

    backend.load_populations({population});
    backend.load_projections({input_projection, loop_projection});
    backend._init();

    auto network = knp::framework::synchronization::get_network_copy(backend);
    ASSERT_EQ(network.get_projections().size(), 2);
    ASSERT_EQ(network.get_populations().size(), 1);
    auto &proj0 = std::get<knp::testing::DeltaProjection>(network.get_projections()[0]);
    auto &proj1 = std::get<knp::testing::DeltaProjection>(network.get_projections()[1]);
    auto &pop = std::get<knp::testing::BLIFATPopulation>(network.get_populations()[0]);
    ASSERT_EQ(proj0.size(), 1);
    ASSERT_EQ(proj1.size(), 1);
    ASSERT_EQ(pop.size(), 1);
}


TEST(SynchronizationSuite, MultiThreadCpuTest)
{
    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.
    knp::testing::MTestingBack backend;
    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};
    auto loop_projection =
        knp::testing::DeltaProjection{population.get_uid(), population.get_uid(), knp::testing::synapse_generator, 1};
    auto input_projection = knp::testing::DeltaProjection{
        knp::core::UID{false}, population.get_uid(), knp::testing::input_projection_gen, 1};

    backend.load_populations({population});
    backend.load_projections({input_projection, loop_projection});
    backend._init();

    auto network = knp::framework::synchronization::get_network_copy(backend);
    ASSERT_EQ(network.get_projections().size(), 2);
    ASSERT_EQ(network.get_populations().size(), 1);
    auto &proj0 = std::get<knp::testing::DeltaProjection>(network.get_projections()[0]);
    auto &proj1 = std::get<knp::testing::DeltaProjection>(network.get_projections()[1]);
    auto &pop = std::get<knp::testing::BLIFATPopulation>(network.get_populations()[0]);
    ASSERT_EQ(proj0.size(), 1);
    ASSERT_EQ(proj1.size(), 1);
    ASSERT_EQ(pop.size(), 1);
}
