/**
 * @file main.cpp
 * @brief Example of running a simple network.
 * @kaspersky_support A. Vartenkov
 * @date 28.03.2024
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

#include <knp/framework/io/out_converters/convert_set.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/synapse-traits/delta.h>

#include <filesystem>


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


int main(int argc, const char *const argv[])
{
    // Creates a population object with one BLIFAT neuron.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235842
    BLIFATPopulation population{neuron_generator, 1};
    // Creates a projection object with one delta synapse, which loops the output of the population
    // to itself.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235844
    DeltaProjection loop_projection = DeltaProjection{population.get_uid(), population.get_uid(), synapse_generator, 1};
    // Creates an input projection object with one delta synapse, which is assigned
    // a null ID (knp::core::UID{false}). The projection receives spikes from
    // the input channel and sends synaptic impact to the population object.
    DeltaProjection input_projection =
        DeltaProjection{knp::core::UID{false}, population.get_uid(), input_projection_gen, 1};

    // Stores the input projection UID in the input_uid variable.
    knp::core::UID input_uid = input_projection.get_uid();
    // Stores the population UID in the output_uid variable.
    knp::core::UID output_uid = population.get_uid();

    // Creates a network object.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
    knp::framework::Network network;

    // Adds the created population object to the neural network object.
    network.add_population(std::move(population));
    // Adds the created input_projection object to the neural network object.
    network.add_projection(std::move(input_projection));
    // Adds the created loop_projection object that loops the
    // population output on itself, into the neural network object.
    network.add_projection(std::move(loop_projection));

    // Creates arbitrary i_channel_uid and o_channel_uid identifiers for the input and output channels.
    // Online Help links:
    // https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=244944
    // https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243539
    knp::core::UID i_channel_uid, o_channel_uid;

    // Creates a model object and passes the neural network object to it.
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235849
    knp::framework::Model model(std::move(network));
    // Passes to the model object the created input channel ID (i_channel_uid)
    // and the input projection ID (input_uid).
    model.add_input_channel(i_channel_uid, input_uid);
    // Passes to the model object the created output channel ID (o_channel_uid)
    // and the population ID (output_uid).
    model.add_output_channel(o_channel_uid, output_uid);

    auto input_gen = [](knp::core::Step step) -> knp::core::messaging::SpikeData
    {
        // Sends spikes at the following neural network execution steps: 0, 5, 10, and 15.
        if (step % 5 == 0)
        {
            knp::core::messaging::SpikeData s;
            s.push_back(0);
            return s;
        }
        return {};
    };

    // Specifies the path to the required dynamic backend library on your computer.
    auto backend_path = std::filesystem::path(argv[0]).parent_path() / "knp-cpu-single-threaded-backend";
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243548
    knp::framework::BackendLoader backend_loader;

    // Creates the model executor object (model_executor). Passes to the model_executor object the model object,
    // the path to the backend (backend_path), the input channel ID (i_channel_uid), and
    // the spike generation functor (input_gen).
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=251296
    knp::framework::ModelExecutor model_executor(
        model, backend_loader.load(backend_path), {{i_channel_uid, input_gen}});

    // Receives a link to the output channel object (out_channel) from
    // the model executor (model_executor) by the output channel ID (o_channel_uid).
    auto &out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);

    // Starts execution of the 20 steps of the model.
    model_executor.start([](size_t step) { return step < 20; });

    // Creates the results vector that contains the indices of the spike steps.
    std::vector<knp::core::Step> results;
    // Updates the output channel.
    const auto &spikes = out_channel.update();
    // Allocates a memory area for spikes.
    results.reserve(spikes.size());

    // Writes the indices of the steps at which the spikes are received by the output channel
    // into the results vector.
    std::transform(
        spikes.cbegin(), spikes.cend(), std::back_inserter(results),
        [](const auto &spike_msg) { return spike_msg.header_.send_time_; });
    // Outputs the space-separated indices of the steps at which spikes are received by the output channel.
    for (const auto &s : results) std::cout << s << " ";
    // Outputs the newline character and calls the flush() method.
    std::cout << std::endl;
}
