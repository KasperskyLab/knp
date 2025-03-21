/**
 * @file evaluation.cpp
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
#include "evaluation.h"

#include <knp/core/messaging/messaging.h>

#include <algorithm>
#include <fstream>


void Target::obtain_output_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices)
{
    for (auto i : firing_neuron_indices) ++prediction_votes_[i % get_num_targets()];
    if (!((tact + 1) % state_duration_))
    {
        int starting_index = index_offset_;
        int j = starting_index;
        int n_max = 0;
        int predicted_state = -1;
        do
        {
            if (prediction_votes_[j] > n_max)
            {
                n_max = prediction_votes_[j];
                predicted_state = j;
            }
            if (++j == get_num_targets()) j = 0;
        } while (j != starting_index);
        if (++index_offset_ == get_num_targets()) index_offset_ = 0;
        predicted_states_.push_back(std::make_pair(predicted_state, n_max));
        if (n_max) max_vote_[predicted_state] = std::max(max_vote_[predicted_state], n_max);
        std::fill(prediction_votes_.begin(), prediction_votes_.end(), 0);
    }
    ++tact;
}


int Target::finalize(enum Criterion criterion, const std::filesystem::path &out_file_path) const
{
    int nExistingClasses, nNoClass;
    if (none_of(max_vote_.begin(), max_vote_.end(), std::identity()))  // No predictions at all...
        return 1;
    std::vector<int> predictions;
    auto v_lims = max_vote_;
    auto v_pars = v_lims;
    auto v_parbest = v_pars;
    int ncorrbest = 0;
    size_t i;
    int curcla = 0;
    do
    {
        predictions.clear();
        for (i = 0; i < predicted_states_.size(); ++i)
            predictions.push_back(
                predicted_states_[i].first == -1 || predicted_states_[i].second < v_pars[predicted_states_[i].first]
                    ? -1
                    : predicted_states_[i].first);
        int ncorr = 0;
        for (i = 0; i < predictions.size(); ++i)
            if (predictions[i] == states_[i]) ++ncorr;
        if (ncorr > ncorrbest)
        {
            ncorrbest = ncorr;
            v_parbest = v_pars;
        }
        if (v_pars[curcla])
            --v_pars[curcla];
        else
        {
            v_pars = v_parbest;
            ++curcla;
        }
    } while (curcla < get_num_targets());

    predictions.clear();
    for (i = 0; i < predicted_states_.size(); ++i)
        predictions.push_back(
            predicted_states_[i].first == -1 || predicted_states_[i].second < v_parbest[predicted_states_[i].first]
                ? -1
                : predicted_states_[i].first);
    int nerr = 0;

    std::vector<Result> vcr(get_num_targets());
    int num_true_negatives = 0;
    for (i = 0; i < predicted_states_.size(); ++i)
    {
        int predicted = predictions[i];
        if (states_[i] != -1) ++vcr[states_[i]].real;
        if (predicted != -1) ++vcr[predicted].predicted;
        if (predicted != states_[i])
            ++nerr;
        else if (predicted != -1)
            ++vcr[predicted].correcty_predicted;
        else
            ++num_true_negatives;
    }
    if (!out_file_path.empty())
    {
        std::ofstream out_stream(out_file_path);
        out_stream << "TARGET,THRESHOLD,PRECISION,RECAL"
                      "L,F\n";
        for (int cla = 0; cla < get_num_targets(); ++cla)
        {
            float precision =
                vcr[cla].predicted ? vcr[cla].correcty_predicted / static_cast<float>(vcr[cla].predicted) : 0.F;

            float recall = vcr[cla].real ? vcr[cla].correcty_predicted / static_cast<float>(vcr[cla].real) : 0.F;
            out_stream << cla << ',' << v_parbest[cla] << ',' << precision << ',' << recall << ','
                       << (precision && recall ? 2 / (1 / precision + 1 / recall) : 0.F) << std::endl;
        }
        for (i = 0; i < predicted_states_.size(); ++i)
            out_stream << states_[i] << ',' << predicted_states_[i].first << ',' << predicted_states_[i].second << ','
                       << predictions[i] << std::endl;
    }
    double dWeightedCorrect;
    switch (criterion)
    {
        case Criterion::absolute_error:
            return static_cast<int>(std::lround(10000.0 * (1 - static_cast<double>(nerr) / predicted_states_.size())));

        case Criterion::weighted_error:
            nExistingClasses = 0;
            dWeightedCorrect = 0.;
            nNoClass = static_cast<int>(predicted_states_.size());
            for (int cla = 0; cla < get_num_targets(); ++cla)
                if (vcr[cla].real)
                {
                    dWeightedCorrect += vcr[cla].correcty_predicted / static_cast<double>(vcr[cla].real);
                    ++nExistingClasses;
                    nNoClass -= vcr[cla].real;
                }
            if (nNoClass)
            {
                ++nExistingClasses;
                dWeightedCorrect += num_true_negatives / static_cast<double>(nNoClass);
            }
            return static_cast<int>(std::lround(10000.0 * dWeightedCorrect / nExistingClasses));

        default:
            return finalize_averaged_f(vcr);
    }
}


int Target::finalize_averaged_f(const std::vector<Result> &vcr) const
{
    double dWeightedF = 0.;
    int ndef = 0;
    for (int target_index = 0; target_index < get_num_targets(); ++target_index)
    {
        if (vcr[target_index].real)
        {
            double precision = vcr[target_index].predicted ? vcr[target_index].correcty_predicted /
                                                                 static_cast<double>(vcr[target_index].predicted)
                                                           : 0.F;
            double recall = vcr[target_index].correcty_predicted / static_cast<double>(vcr[target_index].real);
            double f_measure = precision && recall ? 2 / (1 / precision + 1 / recall) : 0.;
            dWeightedF += f_measure * vcr[target_index].real;
            ndef += vcr[target_index].real;
        }
    }
    return static_cast<int>(std::lround(10000.0 * dWeightedF / ndef));
}
