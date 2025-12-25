/**
 * @file save-load-networks.h
 * @brief Example of saving and loading a simple network.
 * @kaspersky_support A. Vartenkov
 * @date 04.04.2024
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

#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/framework/network.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/synapse-traits/delta.h>


// Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=308225
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
// Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=308213
using BLIFATPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;


// A function is implemented that generates synapses for the projection connected
// to an input channel. The function is stored in the input_projection_gen variable.
inline std::optional<DeltaProjection::Synapse> input_projection_gen(size_t /*index*/)  // NOLINT
{
    return DeltaProjection::Synapse{{1.0, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
}


// A function is implemented that generates synapses for the projection, which loops the
// population output on itself. The function is stored in the synapse_generator variable.
inline std::optional<DeltaProjection::Synapse> synapse_generator(size_t /*index*/)  // NOLINT
{
    return DeltaProjection::Synapse{{1.0, 6, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
}


// A function is implemented that generates neurons.
// The function is stored in the neuron_generator variable.
inline knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> neuron_generator(size_t)  // NOLINT
{
    return knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>{};
}


// Creates a simple neural network with the following structure:
// input channel -> input_projection -> population <=> loop_projection.
// Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
knp::framework::Network make_simple_network()
{
    // Creates a population object with one BLIFAT neuron.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235842
    BLIFATPopulation population{neuron_generator, 1};
    // Creates a projection object with one delta synapse, which loops the output of the population
    // to itself.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235844
    knp::core::Projection<knp::synapse_traits::DeltaSynapse> loop_projection =
        DeltaProjection{population.get_uid(), population.get_uid(), synapse_generator, 1};
    // Creates an input projection object with one delta synapse, which is assigned
    // a null ID (knp::core::UID{false}). The projection receives spikes from
    // the input channel and sends synaptic impact to the population object.
    knp::core::Projection<knp::synapse_traits::DeltaSynapse> input_projection =
        DeltaProjection{knp::core::UID{false}, population.get_uid(), input_projection_gen, 1};
    // Creates a network object.
    knp::framework::Network network;
    // Adds the created population object to the neural network object.
    network.add_population(population);
    // Adds the created input_projection object to the neural network object.
    network.add_projection(input_projection);
    // Adds the created loop_projection object that loops the
    // population output on itself, into the neural network object.
    network.add_projection(loop_projection);
    return network;
}


int main()
{
    auto network = make_simple_network();
    // Saves the neural network in the SONATA format.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=274991
    knp::framework::sonata::save_network(network, ".");
    // Loads the neural network from the SONATA format.
    auto network2 = knp::framework::sonata::load_network(".");
    std::cout << network2.populations_count() << " " << network2.projections_count() << std::endl;
    return EXIT_SUCCESS;
}
