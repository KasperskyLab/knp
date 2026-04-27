/**
 * @file connectivity_validator_test.cpp
 * @brief Connectivity validator test.
 * @kaspersky_support David P.
 * @date 10.04.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <knp/framework/network.h>
#include <knp/framework/network_validation/executor.h>
#include <knp/framework/network_validation/validators/connectivity.h>
#include <knp/framework/population/neuron_parameters_generators.h>

#include <tests_common.h>


TEST(ConnectivityValidator, ErrorCodes)
{
    knp::framework::Network network;

    network.add_population(knp::core::Population<knp::neuron_traits::BLIFATNeuron>(
        knp::core::UID(false),
        knp::framework::population::neurons_generators::make_default<knp::neuron_traits::BLIFATNeuron>(), 0));

    network.add_projection(
        knp::core::Projection<knp::synapse_traits::DeltaSynapse>(knp::core::UID(false), knp::core::UID(false)));

    knp::framework::network_validation::Executor executor;
    executor.add_validator(knp::framework::network_validation::Connectivity());

    auto reports = executor.run_validators(network);
    ASSERT_EQ(reports.size(), 1);
    ASSERT_EQ(reports[0].report_.size(), 2);
    ASSERT_EQ(reports[0].report_[0].severity_, knp::framework::network_validation::IssueSeverity::error);
    ASSERT_EQ(
        reports[0].report_[0].code_.value(),
        knp::framework::network_validation::Connectivity::projection_not_connected);
    ASSERT_EQ(reports[0].report_[1].severity_, knp::framework::network_validation::IssueSeverity::error);
    ASSERT_EQ(
        reports[0].report_[1].code_.value(),
        knp::framework::network_validation::Connectivity::population_not_connected);
}
