/**
 * @file network.cpp
 * @brief Network class implementation.
 * @kaspersky_support Artiom N.
 * @date 22.03.2023
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

#include <knp/framework/network.h>
#include <knp/framework/population/neuron_generators/slicing_generators.h>
#include <knp/framework/projection/synapse_generators/synapse_slicing_generators.h>

#include <spdlog/spdlog.h>

#include <algorithm>

#include <boost/preprocessor.hpp>


namespace knp::framework
{

Network::PopulationIterator Network::begin_populations()
{
    return populations_.begin();
}


Network::PopulationConstIterator Network::begin_populations() const
{
    return populations_.cbegin();
}


Network::PopulationIterator Network::end_populations()
{
    return populations_.end();
}


Network::PopulationConstIterator Network::end_populations() const
{
    return populations_.cend();
}


Network::ProjectionIterator Network::begin_projections()
{
    return projections_.begin();
}


Network::ProjectionConstIterator Network::begin_projections() const
{
    return projections_.cbegin();
}


Network::ProjectionIterator Network::end_projections()
{
    return projections_.end();
}


Network::ProjectionConstIterator Network::end_projections() const
{
    return projections_.cend();
}


template <typename VT>
typename std::vector<VT>::iterator find_variant(const knp::core::UID &uid, std::vector<VT> &container)
{
    auto result = std::find_if(
        container.begin(), container.end(),
        [&uid](VT &p_variant) -> bool
        { return std::visit([&uid](auto &var_val) { return var_val.get_uid() == uid; }, p_variant); });
    return result;
}


template <typename VT>
typename std::vector<VT>::const_iterator find_cvariant(const knp::core::UID &uid, const std::vector<VT> &container)
{
    auto result = std::find_if(
        container.cbegin(), container.cend(),
        [&uid](const VT &p_variant) -> bool
        { return std::visit([&uid](const auto &var_val) { return var_val.get_uid() == uid; }, p_variant); });
    return result;
}


void Network::add_population(core::AllPopulationsVariant &&population)
{
    SPDLOG_DEBUG("Adding population variant...");

    // check_population_constraints(std::visit([](const auto &var_val) { return var_val; }, population));

    populations_.emplace_back(population);
}


template <typename PopulationType>
void Network::add_population(typename std::decay<PopulationType>::type &&population)
{
    SPDLOG_DEBUG("Adding population [move]...");

    check_population_constraints(population);

    add_population(population);
}


template <typename PopulationType>
void Network::add_population(typename std::decay<PopulationType>::type &population)
{
    SPDLOG_DEBUG("Adding population [copy]...");

    check_population_constraints(population);
    add_population(population);
}


bool Network::is_population_exists(const knp::core::UID &population_uid) const
{
    const auto pop_iterator = find_cvariant<core::AllPopulationsVariant>(population_uid, populations_);
    return pop_iterator != populations_.cend();
}


template <typename NeuronType>
knp::core::Population<NeuronType> &Network::get_population(const knp::core::UID &population_uid)
{
    static_assert(
        boost::mp11::mp_contains<AllPopulations, knp::core::Population<NeuronType>>(),
        "This population type is not supported by the Network class. Add type to the population type list.");

    SPDLOG_DEBUG("Getting population {}...", std::string(population_uid));
    if (auto pop_iterator = find_variant<core::AllPopulationsVariant>(population_uid, populations_);
        pop_iterator != populations_.end())
    {
        return std::get<knp::core::Population<NeuronType>>(*pop_iterator);
    }
    throw std::logic_error("Cannot find population with UID \"" + std::string(population_uid) + "\".");
}


template <typename NeuronType>
const knp::core::Population<NeuronType> &Network::get_population(const knp::core::UID &population_uid) const
{
    SPDLOG_DEBUG("Getting population {}...", std::string(population_uid));
    if (const auto pop_iterator = find_cvariant<core::AllPopulationsVariant>(population_uid, populations_);
        pop_iterator != populations_.cend())
    {
        return std::get<knp::core::Population<NeuronType>>(*pop_iterator);
    }
    throw std::logic_error("Cannot find population with UID \"" + std::string(population_uid) + "\".");
}


core::AllPopulationsVariant &Network::get_population(const knp::core::UID &population_uid)
{
    SPDLOG_DEBUG("Getting projection {}...", std::string(population_uid));

    auto iter = find_variant<core::AllPopulationsVariant>(population_uid, populations_);

    if (populations_.end() == iter)
    {
        throw std::logic_error("Cannot find population with UID \"" + std::string(population_uid) + "\".");
    }

    return *iter;
}


void Network::remove_population(const core::UID &population_uid)
{
    SPDLOG_DEBUG("Removing population with UID {}...", std::string(population_uid));
    auto result = find_variant<core::AllPopulationsVariant>(population_uid, populations_);

    if (result == populations_.end())
    {
        throw std::logic_error(
            "Cannot remove non-existent population with UID \"" + std::string(population_uid) + "\".");
    }
    populations_.erase(result);
}


void Network::add_projection(core::AllProjectionsVariant &&projection)
{
    SPDLOG_DEBUG("Adding projection variant...");

    // check_projection_constraints(std::visit([](const auto &var_val) { return var_val; }, projection));
    projections_.emplace_back(projection);
}


template <typename ProjectionType>
void Network::add_projection(typename std::decay<ProjectionType>::type &&projection)
{
    SPDLOG_DEBUG("Adding projection [move] {}...", std::string(projection.get_uid()));

    check_projection_constraints(projection);
    projections_.emplace_back(std::move(projection));
}


template <typename ProjectionType>
void Network::add_projection(typename std::decay<ProjectionType>::type &projection)
{
    SPDLOG_DEBUG("Adding projection [copy] {}...", std::string(projection.get_uid()));

    check_projection_constraints(projection);
    projections_.emplace_back(projection);
}


template <typename SynapseType>
void Network::add_projection(
    knp::core::UID projection_uid, knp::core::UID pre_population_uid, knp::core::UID post_population_uid,
    typename knp::core::Projection<SynapseType>::SynapseGenerator generator, size_t synapses_count)
{
    SPDLOG_DEBUG("Adding projection {}...", std::string(projection_uid));

    auto projection = knp::core::Projection<SynapseType>(
        projection_uid, pre_population_uid, post_population_uid, generator, synapses_count);
    check_projection_constraints(projection);

    add_projection(core::AllProjectionsVariant(std::move(projection)));
}


bool Network::is_projection_exists(const knp::core::UID &projection_uid) const
{
    const auto proj_iterator = find_cvariant<core::AllProjectionsVariant>(projection_uid, projections_);
    return proj_iterator != projections_.cend();
}


template <typename SynapseType>
knp::core::Projection<SynapseType> &Network::get_projection(const knp::core::UID &projection_uid)
{
    static_assert(
        boost::mp11::mp_contains<core::AllProjections, knp::core::Projection<SynapseType>>(),
        "This projection type is not supported by the Network class. Add type to the projection type list.");

    SPDLOG_DEBUG("Getting projection {}...", std::string(projection_uid));

    auto proj_iterator = find_variant<core::AllProjectionsVariant>(projection_uid, projections_);
    if (proj_iterator != projections_.end())
    {
        return std::get<knp::core::Projection<SynapseType>>(*proj_iterator);
    }

    throw std::logic_error("Cannot find projection with UID \"" + std::string(projection_uid) + "\".");
}


template <typename SynapseType>
const knp::core::Projection<SynapseType> &Network::get_projection(const knp::core::UID &projection_uid) const
{
    SPDLOG_DEBUG("Getting projection {}...", std::string(projection_uid));

    auto proj_iterator = find_cvariant<core::AllProjectionsVariant>(projection_uid, projections_);
    if (proj_iterator != projections_.cend())
    {
        return std::get<knp::core::Projection<SynapseType>>(*proj_iterator);
    }

    throw std::logic_error("Cannot find projection with UID \"" + std::string(projection_uid) + "\".");
}


core::AllProjectionsVariant &Network::get_projection(const knp::core::UID &projection_uid)
{
    SPDLOG_DEBUG("Getting projection {}...", std::string(projection_uid));

    auto iter = find_variant<core::AllProjectionsVariant>(projection_uid, projections_);

    if (projections_.end() == iter)
    {
        throw std::logic_error("Cannot find projection with UID \"" + std::string(projection_uid) + "\".");
    }

    return *iter;
}


void Network::remove_projection(const core::UID &projection_uid)
{
    SPDLOG_DEBUG("Removing projection with UID {}", std::string(projection_uid));
    auto result = find_variant<core::AllProjectionsVariant>(projection_uid, projections_);

    if (result == projections_.end())
    {
        throw std::logic_error(
            "Cannot remove non-existent projection with UID \"" + std::string(projection_uid) + "\".");
    }
    projections_.erase(result);
}


template <typename BaseSynapseType>
void upcast_projections(Network &network)
{
    using BaseSynapseParams = synapse_traits::synapse_parameters<BaseSynapseType>;

    for (auto proj_iter = network.begin_projections(); proj_iter != network.end_projections(); ++proj_iter)
    {
        auto &projection = *proj_iter;
        bool is_converted = false;
        auto new_projection = std::visit([&is_converted](const auto &proj)
            {
                using Synapse = typename std::remove_reference_t<decltype(proj)>::ProjectionSynapseType;
                using SynapseParams = synapse_traits::synapse_parameters<Synapse>;

                if constexpr (std::is_base_of_v<BaseSynapseParams, SynapseParams>)
                {
                    auto res = projection::synapse_generators::upcast_projection<synapse_traits::DeltaSynapse, Synapse>(
                            proj);
                    is_converted = true;
                    return core::AllProjectionsVariant{res};
                }
                // TODO: Remove unnecessary copying.
                return core::AllProjectionsVariant{proj};
            }, projection);
        if (is_converted)
        {
            projection = std::move(new_projection);
        }
    }
}


template <typename BaseNeuronType>
void upcast_populations(Network &network)
{
    using BaseNeuronParams = neuron_traits::neuron_parameters<BaseNeuronType>;

    for (auto pop_iter = network.begin_populations(); pop_iter != network.end_populations(); ++pop_iter)
    {
        auto &population = *pop_iter;
        bool is_converted = false;
        core::AllPopulationsVariant new_population = std::visit([&is_converted](auto &pop)
            {
                using NeuronType = typename std::remove_reference_t<decltype(pop)>::PopulationNeuronType;
                using NeuronParams = neuron_traits::neuron_parameters<NeuronType>;
                if constexpr (std::is_base_of_v<BaseNeuronParams, NeuronParams>)
                {
                    is_converted = true;
                    auto res =  population::neurons_generators::upcast_population<BaseNeuronType, NeuronType>(pop);
                    return core::AllPopulationsVariant{res};
                }
                // TODO: Remove unnecessary copying.
                return core::AllPopulationsVariant{pop};
            }, population);
        if (is_converted)
        {
            population = std::move(new_population);
        }
    }
}


template <typename PopulationType>
void Network::check_population_constraints(const PopulationType &population) const
{
    if (is_population_exists(population.get_uid()))
    {
        const std::string msg =
            "Population with UID = " + std::string(population.get_uid()) + " already exists in the network.";
        SPDLOG_ERROR("{}", msg);
        throw std::logic_error(msg);
    }
}


template <typename ProjectionType>
void Network::check_projection_constraints(const ProjectionType &projection) const
{
    const auto proj_uid = projection.get_uid();
    if (is_projection_exists(proj_uid))
    {
        const std::string msg = "Projection with UID = " + std::string(proj_uid) + " already exists in the network.";
        SPDLOG_ERROR("{}", msg);
        throw std::logic_error(msg);
    }

    if (core::UID{false} != projection.get_presynaptic() && is_population_exists(projection.get_presynaptic()))
    {
        SPDLOG_WARN("Presynaptic population with UID = {} doesn't exist.", std::string(projection.get_presynaptic()));
    }

    if (is_population_exists(projection.get_postsynaptic()))
    {
        SPDLOG_WARN("Postsynaptic population with UID = {} doesn't exist.", std::string(projection.get_postsynaptic()));
    }
}


namespace nt = knp::neuron_traits;

#define INSTANCE_POPULATION_FUNCTIONS(n, template_for_instance, neuron_type)                                  \
    template KNP_DECLSPEC void Network::add_population<knp::core::Population<nt::neuron_type>>(               \
        knp::core::Population<nt::neuron_type> &&);                                                           \
    template KNP_DECLSPEC void Network::add_population<knp::core::Population<nt::neuron_type>>(               \
        knp::core::Population<nt::neuron_type> &);                                                            \
    template KNP_DECLSPEC knp::core::Population<nt::neuron_type>                                              \
        &Network::get_population<knp::neuron_traits::neuron_type>(const knp::core::UID &);                    \
    template KNP_DECLSPEC const knp::core::Population<nt::neuron_type>                                        \
        &Network::get_population<knp::neuron_traits::neuron_type>(const knp::core::UID &) const;              \
    template KNP_DECLSPEC void Network::check_population_constraints<knp::core::Population<nt::neuron_type>>( \
        const knp::core::Population<nt::neuron_type> &) const;                                                \
    template KNP_DECLSPEC void upcast_populations<knp::neuron_traits::neuron_type>(Network& network);
namespace st = knp::synapse_traits;

#define INSTANCE_PROJECTION_FUNCTIONS(n, template_for_instance, synapse_type)                                       \
    template KNP_DECLSPEC void Network::add_projection<knp::core::Projection<st::synapse_type>>(                    \
        knp::core::Projection<st::synapse_type> &&);                                                                \
    template KNP_DECLSPEC void Network::add_projection<knp::core::Projection<st::synapse_type>>(                    \
        knp::core::Projection<st::synapse_type> &);                                                                 \
    template KNP_DECLSPEC void Network::add_projection<st::synapse_type>(                                           \
        knp::core::UID, knp::core::UID, knp::core::UID,                                                             \
        typename knp::core::Projection<st::synapse_type>::SynapseGenerator, size_t);                                \
    template KNP_DECLSPEC knp::core::Projection<st::synapse_type> &Network::get_projection<st::synapse_type>(       \
        const knp::core::UID &);                                                                                    \
    template KNP_DECLSPEC const knp::core::Projection<st::synapse_type> &Network::get_projection<st::synapse_type>( \
        const knp::core::UID &) const;                                                                              \
    template KNP_DECLSPEC void Network::check_projection_constraints<knp::core::Projection<st::synapse_type>>(      \
        const knp::core::Projection<st::synapse_type> &) const;                                                     \
    template KNP_DECLSPEC void upcast_projections<st::synapse_type>(Network &network);                                                                                                                    \
// cppcheck-suppress unknownMacro
BOOST_PP_SEQ_FOR_EACH(INSTANCE_POPULATION_FUNCTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_NEURONS))
// cppcheck-suppress unknownMacro
BOOST_PP_SEQ_FOR_EACH(INSTANCE_PROJECTION_FUNCTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_SYNAPSES))

}  // namespace knp::framework
