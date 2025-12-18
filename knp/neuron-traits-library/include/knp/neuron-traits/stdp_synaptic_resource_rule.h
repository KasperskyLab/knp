/**
 * @file stdp_synaptic_resource_rule.h
 * @brief Synaptic resource-based STDP neuron parameters.
 * @kaspersky_support Artiom N.
 * @date 06.10.2023
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

#include <cinttypes>
#include <limits>


/**
 * @brief Namespace for neuron traits.
 */
namespace knp::neuron_traits
{
/**
 * @brief Neuron type with resource-based STDP learning rule.
 * @details You can use this structure as a wrapper over base neuron type.
 * @tparam NeuronType base neuron type.
 */
template <typename NeuronType>
struct SynapticResourceSTDPNeuron;

/**
 * @brief Neuron state in the continuous spike generation period.
 */
enum class ISIPeriodType
{
    /**
     * @brief Neuron is forced, i.e. spiked while receiving an excitatory signal from a non-plastic neuron.
     */
    is_forced,
    /**
     * @brief Neuron receives the first non-forced spike.
     */
    period_started,
    /**
     * @brief Neuron continues to receive non-forced spikes.
     */
    period_continued,
    /**
     * @brief Sequence of spikes has ended. A new sequence has not started yet.
     * @note The state is not commonly used as neuron state changes after a spike.
     */
    not_in_period
};


/**
 * @brief Template for neuron parameters supported by synaptic resource-based STDP.
 * @tparam NeuronType base neuron type without plasticity.
 */
template <typename NeuronType>
struct default_values<SynapticResourceSTDPNeuron<NeuronType>>
{
    /**
     * @brief Base neuron default parameters.
     */
    default_values<NeuronType> neuron_;
};


/**
 * @brief Template for neuron parameters supported by synaptic resource-based STDP.
 * @tparam NeuronType base neuron type without plasticity.
 * @note The class contains all neuron parameters of the base neuron.
 */
template <typename NeuronType>
struct neuron_parameters<SynapticResourceSTDPNeuron<NeuronType>> : public neuron_parameters<NeuronType>
{
    /**
     * @brief Default constructor.
     */
    neuron_parameters() = default;

    /**
     * @brief Construct parameters for synaptic resource-based STDP from a base neuron.
     * @param base_neuron starting parameters of the base neuron type.
     */
    explicit neuron_parameters(const neuron_parameters<NeuronType> &base_neuron)
        : neuron_parameters<NeuronType>(base_neuron)
    {
    }

    /**
     * @brief Coefficient for the `additional_threshold_` neuron parameter.
     * @note Current threshold value for membrane potential is composed from three parameters:
     * activation_threshold_, dynamic_threshold_, and additional_threshold_.
     */
    float synapse_sum_threshold_coefficient_ = 0.0F;


    /**
     * @brief Time parameter for dopamine plasticity.
     */
    uint32_t dopamine_plasticity_time_ = 1;

    /**
     * @brief Free synaptic resource.
     */
    float free_synaptic_resource_ = 1;

    /**
     * @brief Synaptic resource threshold value.
     */
    float synaptic_resource_threshold_ = std::numeric_limits<float>::max();

    /**
     * @brief Synaptic resource divided by `number of synapses + resource drain coefficient`. The parameter value
     * defines the number of silent synapses.
     */
    uint32_t resource_drain_coefficient_ = 0;

    /**
     * @brief Dynamic synapse attribute for stability.
     * @details The stability reflects how well a synapse is trained.
     * The value increases with correct responses from a synapse and decreases in case of incorrect ones.
     * The higher the stability, the smaller the changes in synapse weight associated with plasticity.
     * An increase in stability by 1 decreases synapse weight changes by two times.
     */
    float stability_ = 0;

    /**
     * @brief The parameter defines the stability fluctuation value.
     */
    float stability_change_parameter_ = 0;

    /**
     * @brief This value is added to stability at the beginning of each ISI period.
     */
    float stability_change_at_isi_ = 0;

    /**
     * @brief Time between spikes in the ISI period.
     */
    uint32_t isi_max_ = 1;

    /**
     * @brief Hebbian plasticity value.
     */
    float d_h_ = 1.F;

    /**
     * @brief ISI period status.
     */
    ISIPeriodType isi_status_ = ISIPeriodType::not_in_period;

    /**
     * @brief Last step for a non-forced spike.
     */
    uint64_t last_step_ = 0;

    /**
     * @brief Last step for any spike.
     */
    uint64_t last_spike_step_ = 0;

    /**
     * @brief Step of the last unforced spike. Used to update the dopamine value.
     */
    uint64_t first_isi_spike_ = 0;

    /**
     * @brief `true` if a neuron receives a forced synaptic signal.
     */
    bool is_being_forced_ = false;
};

}  // namespace knp::neuron_traits
