/**
 * @file runner.cpp
 * @brief Network validators runner.
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

#include <knp/framework/network_validation/runner.h>

#include <spdlog/spdlog.h>


namespace knp::framework::network_validation
{

void Runner::add_validator(std::string_view name, PopulationValidator validator)
{
    population_validators_.emplace_back(std::string(name), std::move(validator));
}


void Runner::add_validator(std::string_view name, ProjectionValidator validator)
{
    projection_validators_.emplace_back(std::string(name), std::move(validator));
}


void Runner::add_validator(std::string_view name, NetworkValidator validator)
{
    network_validators_.emplace_back(std::string(name), std::move(validator));
}


void Runner::log_reports(const std::vector<Report>& reports)
{
    for (const auto& report : reports)
    {
        switch (report.severity_)
        {
            case ReportSeverity::info:
                SPDLOG_INFO("[{}] {}", report.code_, report.message_);
                break;
            case ReportSeverity::warning:
                SPDLOG_WARN("[{}] {}", report.code_, report.message_);
                break;
            case ReportSeverity::error:
                SPDLOG_ERROR("[{}] {}", report.code_, report.message_);
                break;
            default:
                throw std::logic_error("Unknown severity level.");
        }
    }
}


std::vector<Runner::ValidatorReport> Runner::run_validators(const Network& network)
{
    std::vector<ValidatorReport> reports;

    if (population_validators_.size())
    {
        SPDLOG_INFO("Starting population validators.");
        for (auto& validator : population_validators_)
        {
            SPDLOG_INFO("Running \"{}\".", validator.first);
            std::vector<Report> validator_report;
            for (const auto& pop : network.get_populations())
            {
                auto current_reports = validator.second(pop);
                log_reports(current_reports);
                validator_report.insert(
                    validator_report.end(), std::make_move_iterator(current_reports.begin()),
                    std::make_move_iterator(current_reports.end()));
            }
            reports.push_back({validator.first, std::move(validator_report)});
        }
        SPDLOG_INFO("Finished population validators.");
    }

    if (projection_validators_.size())
    {
        SPDLOG_INFO("Starting projection validators.");
        for (auto& validator : projection_validators_)
        {
            SPDLOG_INFO("Running \"{}\".", validator.first);
            std::vector<Report> validator_report;
            for (const auto& proj : network.get_projections())
            {
                auto current_reports = validator.second(proj);
                log_reports(current_reports);
                validator_report.insert(
                    validator_report.end(), std::make_move_iterator(current_reports.begin()),
                    std::make_move_iterator(current_reports.end()));
            }
            reports.push_back({validator.first, std::move(validator_report)});
        }
        SPDLOG_INFO("Finished projection validators.");
    }

    if (network_validators_.size())
    {
        SPDLOG_INFO("Starting network validators.");
        for (auto& validator : network_validators_)
        {
            SPDLOG_INFO("Running \"{}\".", validator.first);
            auto current_reports = validator.second(network);
            log_reports(current_reports);
            reports.push_back({validator.first, current_reports});
        }
        SPDLOG_INFO("Finished network validators.");
    }

    return reports;
}

}  // namespace knp::framework::network_validation
