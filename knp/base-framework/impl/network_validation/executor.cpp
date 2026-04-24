/**
 * @file executor.cpp
 * @brief Network validators executor.
 * @kaspersky_support David P.
 * @date 08.04.2026
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

#include <knp/framework/network_validation/executor.h>
#include <knp/framework/tags/name.h>

#include <spdlog/spdlog.h>


namespace knp::framework::network_validation
{

Executor::ValidatorUID Executor::add_validator(std::string name, PopulationValidator validator)
{
    knp::core::UID uid;
    population_validators_[uid] = {std::move(name), std::move(validator)};
    return uid;
}


Executor::ValidatorUID Executor::add_validator(PopulationValidator validator)
{
    knp::core::UID uid;
    population_validators_[uid] = {
        "Populations validator #" + std::to_string(population_validators_.size()), std::move(validator)};
    return uid;
}


Executor::ValidatorUID Executor::add_validator(std::string name, ProjectionValidator validator)
{
    knp::core::UID uid;
    projection_validators_[uid] = {std::move(name), std::move(validator)};
    return uid;
}


Executor::ValidatorUID Executor::add_validator(ProjectionValidator validator)
{
    knp::core::UID uid;
    projection_validators_[uid] = {
        "Projections validator #" + std::to_string(projection_validators_.size()), std::move(validator)};
    return uid;
}


Executor::ValidatorUID Executor::add_validator(std::string name, NetworkValidator validator)
{
    knp::core::UID uid;
    network_validators_[uid] = {std::move(name), std::move(validator)};
    return uid;
}


Executor::ValidatorUID Executor::add_validator(NetworkValidator validator)
{
    knp::core::UID uid;
    network_validators_[uid] = {
        "Network validator #" + std::to_string(network_validators_.size()), std::move(validator)};
    return uid;
}


void Executor::log_report(const Report& report)
{
    for (const auto& issue : report)
    {
        switch (issue.severity_)
        {
            case IssueSeverity::info:
                SPDLOG_INFO("[{}] {}", issue.code_.value(), issue.message_);
                break;
            case IssueSeverity::warning:
                SPDLOG_WARN("[{}] {}", issue.code_.value(), issue.message_);
                break;
            case IssueSeverity::error:
                SPDLOG_ERROR("[{}] {}", issue.code_.value(), issue.message_);
                break;
            default:
                throw std::logic_error("Unknown severity level.");
        }
    }
}


std::vector<Executor::ValidatorResult> Executor::run_validators(const Network& network)
{
    std::vector<ValidatorResult> reports;

    if (population_validators_.size())
    {
        SPDLOG_INFO("Starting population validators.");
        for (auto& validator : population_validators_)
        {
            for (const auto& pop : network.get_populations())
            {
                SPDLOG_INFO(
                    "Running \"{}\" on population \"{}\".", std::string(validator.second.first),
                    std::visit([](auto&& pop) { return knp::framework::tags::get_name(pop); }, pop));
                auto report = validator.second.second(pop);
                log_report(report);
                reports.push_back({validator.first, validator.second.first, std::move(report)});
            }
        }
        SPDLOG_INFO("Finished population validators.");
    }

    if (projection_validators_.size())
    {
        SPDLOG_INFO("Starting projection validators.");
        for (auto& validator : projection_validators_)
        {
            for (const auto& proj : network.get_projections())
            {
                SPDLOG_INFO(
                    "Running \"{}\" on projection \"{}\".", std::string(validator.second.first),
                    std::visit([](auto&& proj) { return knp::framework::tags::get_name(proj); }, proj));
                auto report = validator.second.second(proj);
                log_report(report);
                reports.push_back({validator.first, validator.second.first, std::move(report)});
            }
        }
        SPDLOG_INFO("Finished projection validators.");
    }

    if (network_validators_.size())
    {
        SPDLOG_INFO("Starting network validators.");
        for (auto& validator : network_validators_)
        {
            SPDLOG_INFO("Running \"{}\".", std::string(validator.second.first));
            auto report = validator.second.second(network);
            log_report(report);
            reports.push_back({validator.first, validator.second.first, std::move(report)});
        }
        SPDLOG_INFO("Finished network validators.");
    }

    return reports;
}

}  // namespace knp::framework::network_validation
