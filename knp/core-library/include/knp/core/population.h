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

#include <knp/core/cell_traits.h>
#include <knp/core/messaging/synaptic_impact_message.h>
#include <knp/core/types.h>

#include <functional>
#include <optional>
#include <utility>
#include <vector>

/**
 * @brief Core library namespace.
 */
namespace knp::core
{

/**
 * @brief The Population class is a container of cells of the same model.
 *
 * @tparam NeuronType neuron or astrocyte type.
 *
 * @see AllCells.
 */
template <typename NeuronType>
class Population final
{
public:
    /**
     * @brief Type of the population cells.
     */
    using PopulationNeuronType = NeuronType;
    /**
     * @brief Population of cells with the specified cell type.
     */
    using PopulationType = Population<NeuronType>;
    /**
     * @brief Cell parameters selected from the neuron or astrocyte family.
     */
    using NeuronParameters = typename CellTraits<NeuronType>::Parameters;

    /**
     * @brief Type of the cell generator.
     *
     * @param index current cell index.
     */
    using NeuronGenerator = std::function<std::optional<NeuronParameters>(size_t index)>;

    /**
     * @brief Type of the container that contains cells.
     */
    using NeuronsContainer = std::vector<NeuronParameters>;

    /**
     * @brief Iterator for cells.
     */
    using iterator = typename NeuronsContainer::iterator;

public:  // NOLINT
    /**
     * @brief Construct a population by running a cell generator.
     *
     * @param generator cell generator.
     * @param neurons_count number of times to run the cell generator.
     */
    Population(NeuronGenerator generator, size_t neurons_count);

    /**
     * @brief Construct a population by running a cell generator.
     *
     * @param uid population UID.
     * @param generator cell generator.
     * @param neurons_count number of times to run the cell generator.
     */
    Population(const knp::core::UID &uid, NeuronGenerator generator, size_t neurons_count);

public:  // NOLINT
    /**
     * @brief Get population UID.
     *
     * @return population UID.
     */
    [[nodiscard]] const UID &get_uid() const { return base_.uid_; }

    /**
     * @brief Get tags used by the population.
     *
     * @return population tag map.
     *
     * @see TagMap.
     */
    [[nodiscard]] auto &get_tags() { return base_.tags_; }

    /**
     * @brief Get tags used by the population.
     *
     * @return population tag map.
     *
     * @see TagMap.
     */
    [[nodiscard]] const auto &get_tags() const { return base_.tags_; }

public:  // NOLINT
    /**
     * @brief Get parameters of all cells in the population.
     *
     * @return vector of cell parameters.
     */
    [[nodiscard]] const std::vector<NeuronParameters> &get_neurons_parameters() const { return neurons_; }

    /**
     * @brief Get parameters of the specific cell in the population.
     *
     * @param index index of the population cell.
     *
     * @return specific cell parameters.
     */
    [[nodiscard]] const NeuronParameters &get_neuron_parameters(size_t index) const { return neurons_[index]; }

    /**
     * @brief Set parameters for the specific cell in the population.
     *
     * @param index index of the population cell.
     * @param parameters cell parameters.
     */
    void set_neuron_parameters(size_t index, const NeuronParameters &parameters) { neurons_[index] = parameters; }

    /**
     * @brief Set parameters for all cells in the population.
     *
     * @param parameters vector of cell parameters.
     *
     * @note Copy method.
     */
    void set_neurons_parameters(const std::vector<NeuronParameters> &parameters) { neurons_ = parameters; }

    /**
     * @brief Set parameters for all cells in the population.
     *
     * @param parameters vector of cell parameters.
     *
     * @note Move method.
     */
    void set_neurons_parameters(std::vector<NeuronParameters> &&parameters) { neurons_ = std::move(parameters); }

public:  // NOLINT
    /**
     * @brief Get tags used by cell with the specified index.
     *
     * @param index index of the population cell.
     *
     * @return tag map of the cell.
     */
    [[nodiscard]] TagMap &get_neuron_tags(size_t index)
    {
        return base_.tags_.template get_tag<std::vector<TagMap>>("neuron_tags")[index];
    }

public:  // NOLINT
    /**
     * @brief Add cells to the population.
     *
     * @param generator generator of cell parameters; an empty optional skips the current index.
     * @param count number of times to run the cell generator.
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
                neurons_.template emplace_back<>(std::move(neuron.value()));
#endif
            }
        }
    }

    /**
     * @brief Remove cells with given indexes from the population.
     *
     * @param neuron_indexes indexes of cells to remove.
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
     * @brief Remove a specific cell from the population.
     *
     * @param neuron_index index of the cell to remove.
     */
    void remove_neuron(const size_t &neuron_index)
    {
        auto iter = neurons_.begin();
        std::advance(iter, neuron_index);
        neurons_.erase(iter);
    }

public:  // NOLINT
    /**
     * @brief Get parameter values of a cell with the given index.
     *
     * @param index cell index.
     * @return cell parameters.
     *
     * @note Constant method.
     */
    const NeuronParameters &operator[](size_t index) const { return get_neuron_parameters(index); }
    /**
     * @brief Get parameter values of a cell with the given index.
     *
     * @param index cell index.
     *
     * @return cell parameters.
     */
    NeuronParameters &operator[](size_t index) { return neurons_[index]; }

    /**
     * @brief Get an iterator pointing to the first element of the population.
     *
     * @return constant population iterator.
     */
    auto begin() const { return neurons_.cbegin(); }
    /**
     * @brief Get an iterator pointing to the first element of the population.
     *
     * @return population iterator.
     */
    auto begin() { return neurons_.begin(); }
    /**
     * @brief Get an iterator pointing to the last element of the population.
     *
     * @return constant iterator.
     */
    auto end() const { return neurons_.cend(); }
    /**
     * @brief Get an iterator pointing to the last element of the population.
     *
     * @return iterator.
     */
    auto end() { return neurons_.end(); }

public:  // NOLINT
    /**
     * @brief Count number of cells in the population.
     *
     * @return number of cells.
     */
    [[nodiscard]] size_t size() const { return neurons_.size(); }

private:
    BaseData base_;
    NeuronsContainer neurons_;
};


/**
 * @brief Population types for all registered neurons and astrocytes in AllCells.
 */
using AllPopulations = boost::mp11::mp_transform<knp::core::Population, AllCells>;

/**
 * @brief Population variant that contains any population type specified in `AllPopulations`.
 *
 * @details Each alternative is a Population<Model>, following the model order in AllCells.
 * Both neuron and astrocyte populations are included.
 *
 * @see AllCells.
 */
using AllPopulationsVariant = boost::mp11::mp_rename<AllPopulations, std::variant>;

}  // namespace knp::core
