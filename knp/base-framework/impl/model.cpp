/**
 * @file model.cpp
 * @brief Model class implementation.
 * @kaspersky_support Artiom N.
 * @date 30.03.2023
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

#include <knp/framework/model.h>

#include <spdlog/spdlog.h>


namespace knp::framework
{

void Model::add_input_channel(const core::UID &channel_uid, const core::UID &projection_uid)
{
    if (!network_.is_projection_exists(projection_uid))
    {
        throw std::logic_error("Projection with UID = " + std::string(projection_uid) + " doesn't exist.");
    }
    in_channels_.insert(decltype(in_channels_)::value_type(channel_uid, projection_uid));
}


void Model::add_output_channel(const core::UID &channel_uid, const core::UID &population_uid)
{
    if (!network_.is_population_exists(population_uid))
    {
        throw std::logic_error("Population with UID = " + std::string(population_uid) + " doesn't exist.");
    }

    out_channels_.insert(decltype(out_channels_)::value_type(channel_uid, population_uid));
}


const std::unordered_multimap<core::UID, core::UID, core::uid_hash> &Model::get_input_channels() const
{
    return in_channels_;
}


const std::unordered_multimap<core::UID, core::UID, core::uid_hash> &Model::get_output_channels() const
{
    return out_channels_;
}


namespace nt = knp::neuron_traits;

#define INSTANCE_POPULATION_FUNCTIONS(n, template_for_instance, neuron_type)      \
    template KNP_DECLSPEC void Model::connect_output_population<nt::neuron_type>( \
        const core::UID &, const core::Population<nt::neuron_type> &);

#define INSTANCE_ASTROCYTE_FUNCTIONS(n, template_for_instance, neuron_type)                          \
    template KNP_DECLSPEC void Model::connect_output_population<knp::astrocyte_traits::neuron_type>( \
        const core::UID &, const core::Population<knp::astrocyte_traits::neuron_type> &);

namespace st = knp::synapse_traits;

#define INSTANCE_PROJECTION_FUNCTIONS(n, template_for_instance, synapse_type)     \
    template KNP_DECLSPEC void Model::connect_input_projection<st::synapse_type>( \
        const core::UID &, const core::Projection<st::synapse_type> &);

// cppcheck-suppress unknownMacro
BOOST_PP_SEQ_FOR_EACH(INSTANCE_POPULATION_FUNCTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_NEURONS))
BOOST_PP_SEQ_FOR_EACH(INSTANCE_ASTROCYTE_FUNCTIONS, , BOOST_PP_VARIADIC_TO_SEQ(ALL_ASTROCYTES))
// cppcheck-suppress unknownMacro
BOOST_PP_SEQ_FOR_EACH(INSTANCE_PROJECTION_FUNCTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_SYNAPSES))

}  // namespace knp::framework
