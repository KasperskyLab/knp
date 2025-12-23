/**
 * @file population.h
 * @brief General population interface.
 * @kaspersky_support Artiom N.
 * @date 18.01.2023
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#include <knp/core/core.h>
#include <knp/core/messaging/synaptic_impact_message.h>
#include <knp/core/uid.h>
#include <knp/neuron-traits/all_traits.h>

#include <functional>
#include <utility>
#include <vector>

/**
 * @brief Core library namespace.
 */
namespace knp::core
{

/**
 * @brief The Population class is a container of neurons of the same model.
 * @tparam NeuronType type of the population neurons.
 * @see ALL_NEURONS.
 */
template <typename NeuronType>
class Population final
{
public:
    /**
     * @brief Type of the population neurons.
     */
    using PopulationNeuronType = NeuronType;
    /**
     * @brief Population of neurons with the specified neuron type.
     */
    using PopulationType = Population<NeuronType>;
    /**
     * @brief Neuron parameters and their values for the specified neuron type.
     */
    using NeuronParameters = neuron_traits::neuron_parameters<NeuronType>;

    /**
     * @brief Type of the neuron generator.
     * @param index current neuron index.
     */
    using NeuronGenerator = std::function<std::optional<NeuronParameters>(size_t index)>;

    /**
     * @brief Type of the container that contains neurons.
     */
    using NeuronsContainer = std::vector<NeuronParameters>;

    /**
     * @brief Iterator for neurons.
     */
    using iterator = typename NeuronsContainer::iterator;

public:  // NOLINT
    /**
     * @brief Construct a population by running a neuron generator.
     * @param generator neuron generator.
     * @param neurons_count number of times to run the neuron generator.
     */
    Population(NeuronGenerator generator, size_t neurons_count);

    /**
     * @brief Construct a population by running a neuron generator.
     * @param uid population UID.
     * @param generator neuron generator.
     * @param neurons_count number of times to run the neuron generator.
     */
    Population(const knp::core::UID &uid, NeuronGenerator generator, size_t neurons_count);

public:  // NOLINT
    /**
     * @brief Get population UID.
     * @return population UID.
     */
    [[nodiscard]] const UID &get_uid() const { return base_.uid_; }

    /**
     * @brief Get tags used by the population.
     * @return population tag map.
     * @see TagMap.
     */
    [[nodiscard]] auto &get_tags() { return base_.tags_; }

    /**
     * @brief Get tags used by the population.
     * @return population tag map.
     * @see TagMap.
     */
    [[nodiscard]] const auto &get_tags() const { return base_.tags_; }

public:  // NOLINT
    /**
     * @brief Get parameters of all neurons in the population.
     * @return vector of neuron parameters.
     */
    [[nodiscard]] const std::vector<NeuronParameters> &get_neurons_parameters() const { return neurons_; }

    /**
     * @brief Get parameters of the specific neuron in the population.
     * @param index index of the population neuron.
     * @return specific neuron parameters.
     */
    [[nodiscard]] const NeuronParameters &get_neuron_parameters(size_t index) const { return neurons_[index]; }

    /**
     * @brief Set parameters for the specific neuron in the population.
     * @param index index of the population neuron.
     * @param parameters neuron parameters.
     */
    void set_neuron_parameters(size_t index, const NeuronParameters &parameters) { neurons_[index] = parameters; }

    /**
     * @brief Set parameters for all neurons in the population.
     * @param parameters vector of neuron parameters.
     * @note Copy method.
     */
    void set_neurons_parameters(const std::vector<NeuronParameters> &parameters) { neurons_ = parameters; }

    /**
     * @brief Set parameters for all neurons in the population.
     * @param parameters vector of neuron parameters.
     * @note Move method.
     */
    void set_neurons_parameters(std::vector<NeuronParameters> &&parameters) { neurons_ = std::move(parameters); }

public:  // NOLINT
    /**
     * @brief Get tags used by neuron with the specified index.
     * @param index index of the population neuron.
     * @return tag map of the neuron.
     */
    [[nodiscard]] TagMap &get_neuron_tags(size_t index)
    {
        return base_.tags_.template get_tag<std::vector<TagMap>>("neuron_tags")[index];
    }

public:  // NOLINT
    /**
     * @brief Add neurons to the population.
     * @param generator type of the neuron generator.
     * @param count number of times to run the neuron generator.
     */
    void add_neurons(NeuronGenerator generator, size_t count)
    {
        neurons_.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            auto neuron = generator(i);
            if (neuron.has_value())
            {
#if defined(_MSC_VER)
                neurons_.emplace_back(std::move(neuron.value()));
#else
                neurons_.template emplace_back(std::move(neuron.value()));
#endif
            }
        }
    }

    /**
     * @brief Remove neurons with given indexes from the population.
     * @param neuron_indexes indexes of neurons to remove.
     */
    void remove_neurons(const std::vector<size_t> &neuron_indexes)
    {
        size_t index_shift = 0;
        for (const auto &index : neuron_indexes)
        {
            remove_neuron(index - index_shift++);
        }
    }

    /**
     * @brief Remove a specific neuron from the population.
     * @param neuron_index index of the neuron to remove.
     */
    void remove_neuron(const size_t &neuron_index)
    {
        auto iter = neurons_.begin();
        std::advance(iter, neuron_index);
        neurons_.erase(iter);
    }

public:  // NOLINT
    /**
     * @brief Get parameter values of a neuron with the given index.
     * @note Constant method.
     * @param index neuron index.
     * @return neuron parameters.
     */
    const NeuronParameters &operator[](size_t index) const { return get_neuron_parameters(index); }
    /**
     * @brief Get parameter values of a neuron with the given index.
     * @param index neuron index.
     * @return neuron parameters.
     */
    NeuronParameters &operator[](size_t index) { return neurons_[index]; }

    /**
     * @brief Get an iterator pointing to the first element of the population.
     * @return constant population iterator.
     */
    auto begin() const { return neurons_.cbegin(); }
    /**
     * @brief Get an iterator pointing to the first element of the population.
     * @return population iterator.
     */
    auto begin() { return neurons_.begin(); }
    /**
     * @brief Get an iterator pointing to the last element of the population.
     * @return constant iterator.
     */
    auto end() const { return neurons_.cend(); }
    /**
     * @brief Get an iterator pointing to the last element of the population.
     * @return iterator.
     */
    auto end() { return neurons_.end(); }

public:  // NOLINT
    /**
     * @brief Count number of neurons in the population.
     * @return number of neurons.
     */
    [[nodiscard]] size_t size() const { return neurons_.size(); }

private:
    BaseData base_;
    NeuronsContainer neurons_;
};


/**
 * @brief List of population types based on neuron types specified in `knp::neuron_traits::AllNeurons`.
 * @details `AllPopulations` takes the value of `Population<NeuronType_1>, Population<NeuronType_2>, ...,
 * Population<NeuronType_n>`, where `NeuronType_[1..n]` is the neuron type specified in
 * `knp::neuron_traits::AllNeurons`. \n For example, if `knp::neuron_traits::AllNeurons` contains BLIFATNeuron and
 * IzhikevichNeuron types, then `AllPopulations` = `Population<BLIFATNeuron>, Population<IzhikevichNeuron>`.
 */
using AllPopulations = boost::mp11::mp_transform<knp::core::Population, knp::neuron_traits::AllNeurons>;

/**
 * @brief Population variant that contains any population type specified in `AllPopulations`.
 * @details `AllPopulationVariants` takes the value of `std::variant<PopulationType_1,..., PopulationType_n>`, where
 * `PopulationType_[1..n]` is the population type specified in `AllPopulations`. \n For example, if `AllPopulations`
 * contains BLIFATNeuron and IzhikevichNeuron types, then `AllPopulationVariants = std::variant<BLIFATNeuron,
 * IzhikevichNeuron>`. \n `AllPopulationVariants` retains the same order of message types as defined in
 * `AllPopulations`.
 * @see ALL_NEURONS.
 */
using AllPopulationsVariant = boost::mp11::mp_rename<AllPopulations, std::variant>;

}  // namespace knp::core
