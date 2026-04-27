/**
 * @file validators_executor_test.cpp
 * @brief Validators executor test.
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
#include <knp/framework/population/neuron_parameters_generators.h>

#include <tests_common.h>

namespace
{
class TestErrorCategory : public std::error_category
{
    [[nodiscard]] const char* name() const noexcept override { return "TestValidatorError"; }
    [[nodiscard]] std::string message(int error) const override { return "Error"; }
};


const std::error_category& error_category() noexcept
{
    static TestErrorCategory instance;
    return instance;
}
}  // namespace


TEST(ValidatorsExecutor, RunValidators)
{
    knp::framework::Network network;
    network.add_projection(
        knp::core::Projection<knp::synapse_traits::DeltaSynapse>(knp::core::UID(false), knp::core::UID(false)));
    network.add_population(knp::core::Population<knp::neuron_traits::BLIFATNeuron>(
        knp::framework::population::neurons_generators::make_default<knp::neuron_traits::BLIFATNeuron>(), 0));

    knp::framework::network_validation::Executor executor;
    executor.add_validator(
        "test0",
        [](const knp::framework::Network&) -> knp::framework::network_validation::Report {
            return {{knp::framework::network_validation::IssueSeverity::info, "message0", {0, error_category()}}};
        });
    executor.add_validator(
        "test1",
        [](const knp::framework::Network::AllProjectionVariants&) -> knp::framework::network_validation::Report {
            return {{knp::framework::network_validation::IssueSeverity::warning, "message1", {1, error_category()}}};
        });
    executor.add_validator(
        "test2",
        [](const knp::framework::Network::AllPopulationVariants&) -> knp::framework::network_validation::Report {
            return {{knp::framework::network_validation::IssueSeverity::error, "message2", {2, error_category()}}};
        });

    auto reports = executor.run_validators(network);
    ASSERT_EQ(reports.size(), 3);
    ASSERT_EQ(reports[0].validator_name_, "test2");
    ASSERT_EQ(reports[0].report_.size(), 1);
    ASSERT_EQ(reports[0].report_[0].severity_, knp::framework::network_validation::IssueSeverity::error);
    ASSERT_EQ(reports[0].report_[0].message_, "message2");
    ASSERT_EQ(reports[0].report_[0].code_.value(), 2);
    ASSERT_EQ(reports[1].validator_name_, "test1");
    ASSERT_EQ(reports[1].report_.size(), 1);
    ASSERT_EQ(reports[1].report_[0].severity_, knp::framework::network_validation::IssueSeverity::warning);
    ASSERT_EQ(reports[1].report_[0].message_, "message1");
    ASSERT_EQ(reports[1].report_[0].code_.value(), 1);
    ASSERT_EQ(reports[2].validator_name_, "test0");
    ASSERT_EQ(reports[2].report_.size(), 1);
    ASSERT_EQ(reports[2].report_[0].severity_, knp::framework::network_validation::IssueSeverity::info);
    ASSERT_EQ(reports[2].report_[0].message_, "message0");
    ASSERT_EQ(reports[2].report_[0].code_.value(), 0);
}
