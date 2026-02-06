/**
 * @file network_constructor.h
 * @brief Functions for network construction.
 * @kaspersky_support D. Postnikov
 * @date 05.02.2026
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
#include <annotated_network.h>

#include <list>
#include <string>


class NetworkConstructor
{
public:
    explicit NetworkConstructor(AnnotatedNetwork &network) : network_(network) {}

    enum PopulationRole
    {
        OUTPUT,
        INPUT,
        NORMAL,
        CHANNELED,
    };

    struct PopulationInfo
    {
        PopulationRole role_;
        bool keep_in_inference_;
        size_t neurons_amount_;
        knp::core::UID uid_;
        std::string name_;
    };

    template <typename Neuron>
    [[nodiscard]] const PopulationInfo &add_population(
        const knp::neuron_traits::neuron_parameters<Neuron> &neuron, size_t neurons_amount, PopulationRole role,
        bool keep_in_inference, const std::string &name)
    {
        PopulationInfo pop_info{role, keep_in_inference, neurons_amount, {}, name};
        network_.network_.add_population(knp::core::Population<Neuron>(
            pop_info.uid_, [&neuron](size_t index) { return neuron; }, pop_info.neurons_amount_));
        network_.data_.population_names_[pop_info.uid_] = pop_info.name_;
        if (pop_info.keep_in_inference_) network_.data_.inference_population_uids_.insert(pop_info.uid_);
        if (OUTPUT == pop_info.role_) network_.data_.output_uids_.push_back(pop_info.uid_);
        return pops_.emplace_back(pop_info);
    }

    [[nodiscard]] const PopulationInfo &add_channeled_population(size_t neurons_amount, bool keep_in_inference)
    {
        return pops_.emplace_back(
            PopulationInfo{CHANNELED, keep_in_inference, neurons_amount, knp::core::UID(false), ""});
    }

    template <typename Synapse, typename Creator>
    knp::core::UID add_projection(
        const knp::synapse_traits::synapse_parameters<Synapse> &synapse, Creator creator, const PopulationInfo &pop_pre,
        const PopulationInfo &pop_post, bool trainable, bool have_wta)
    {
        knp::core::Projection<Synapse> projection = creator(
            have_wta ? knp::core::UID(false) : pop_pre.uid_, pop_post.uid_, pop_pre.neurons_amount_,
            pop_post.neurons_amount_, [&synapse](size_t, size_t) { return synapse; });

        if (trainable) projection.unlock_weights();
        network_.network_.add_projection(projection);
        if (pop_pre.keep_in_inference_ && pop_post.keep_in_inference_)
            network_.data_.inference_internal_projection_.insert(projection.get_uid());
        return projection.get_uid();
    }

private:
    std::list<PopulationInfo> pops_;
    AnnotatedNetwork &network_;
};
