/**
 * @file evaluation.h
 * @brief Functions for network quality estimation.
 * @kaspersky_support A. Vartenkov
 * @date 30.08.2024
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

#pragma once
#include <knp/core/messaging/messaging.h>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>


/**
 * @brief A class used for accuracy evaluation.
 */
class Target
{
public:
    enum class Criterion
    {
        absolute_error,
        weighted_error,
        averaged_f
    };

    Target(int num_target_classes, const std::vector<int> &classes)
        : prediction_votes_(num_target_classes, 0), states_(classes), max_vote_(num_target_classes, 0)
    {
    }

    void obtain_output_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices);

    [[nodiscard]] int get_num_targets() const { return static_cast<int>(prediction_votes_.size()); }

    [[nodiscard]] int finalize(
        enum Criterion criterion = Criterion::absolute_error,
        const std::filesystem::path &strPredictionFile = "") const;

private:
    struct Result
    {
        int real = 0;
        int predicted = 0;
        int correcty_predicted = 0;
    };

    int finalize_absolute_err(const std::vector<Result> &vcr) const;
    int finalize_weighted_err(const std::vector<Result> &vcr) const;
    int finalize_averaged_f(const std::vector<Result> &vcr) const;

    struct TargetClass
    {
        std::string str;
    };

    const std::vector<int> &states_;
    std::vector<std::pair<int, int>> predicted_states_;
    size_t tact = 0;
    const int state_duration_ = 20;
    std::string prediction_file_;
    std::vector<double> possible_predictions_;
    std::vector<int> prediction_votes_;
    std::vector<int> max_vote_;
    std::vector<TargetClass> vtc_;
    int index_offset_ = 0;
};
