/**
 * @file model_monitoring_test.cpp
 * @brief Single-threaded backend test.
 * @kaspersky_support Vartenkov An.
 * @date 07.04.2023
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

#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/model_monitoring.h>

#include <generators.h>
#include <tests_common.h>


TEST(AggregatedSpikesLogger, ModelMonitoring)
{
    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};

    {  //stop spikes from happening
        const auto& params = population.get_neurons_parameters();
        for (size_t i = 0; i < params.size(); i++)
        {
            auto param = params[i];
            param.activation_threshold_ = std::numeric_limits<double>::max();
            //todo change this to set_neuron_parameters when #69 gets fixed
            population.set_neurons_parameters(i, param);
        }
    }

    knp::testing::DeltaProjection input_projection = knp::testing::DeltaProjection{
        knp::core::UID{false}, population.get_uid(), knp::testing::input_projection_gen, 1};
    knp::framework::Network network;
    network.add_population(std::move(population));
    network.add_projection<knp::testing::DeltaProjection>(std::move(input_projection));

    const knp::core::UID i_channel_uid, o_channel_uid;

    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_projection.get_uid());
    model.add_output_channel(o_channel_uid, population.get_uid());

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(
        model, backend_loader.load(knp::testing::get_backend_path()),
        {{i_channel_uid,
          [](knp::core::Step step) -> knp::core::messaging::SpikeData
          {
              if (step % 2 == 0)
              {
                  knp::core::messaging::SpikeData spike_data;
                  spike_data.push_back(0);
                  return spike_data;
              }
              return {};
          }}});

    size_t current_index = 0;
    std::map<std::string, size_t> spike_accumulator;
    std::ostringstream projection_weights_stream;

    knp::framework::monitoring::model_monitoring::add_aggregated_spikes_logger(
        model, {{i_channel_uid, "INPUT"}}, model_executor, current_index, spike_accumulator, projection_weights_stream,
        1);
    model_executor.start([](size_t step) -> bool { return step < 3; });

    ASSERT_STREQ(projection_weights_stream.str().c_str(), "Index, INPUT\n1, 1\n2, 0\n");
}


TEST(ProjectionWeightsLogger, ModelMonitoring)
{
    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};

    knp::testing::ResourceSynapseParams default_synapse;

    knp::testing::ResourceDeltaProjection input_projection{
        knp::core::UID{false}, population.get_uid(),
        [&](size_t index) {
            return knp::testing::ResourceSynapseData{default_synapse, 0, 1};
        },
        1};

    knp::framework::Network network;
    network.add_population(std::move(population));
    network.add_projection<knp::testing::ResourceDeltaProjection>(std::move(input_projection));

    const knp::core::UID i_channel_uid, o_channel_uid;

    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_projection.get_uid());
    model.add_output_channel(o_channel_uid, population.get_uid());

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(
        model, backend_loader.load(knp::testing::get_backend_path()),
        {{i_channel_uid,
          [](knp::core::Step step) -> knp::core::messaging::SpikeData
          {
              knp::core::messaging::SpikeData spike_data;
              spike_data.push_back(0);
              return spike_data;
          }}});

    std::ostringstream projection_weights_stream;
    knp::framework::monitoring::model_monitoring::add_projection_weights_logger(
        projection_weights_stream, model_executor, input_projection.get_uid(), 1);
    model_executor.start([](size_t step) -> bool { return step < 2; });

    ASSERT_STREQ(projection_weights_stream.str().c_str(), "Step: 1\n\nNeuron 1\n0|0 \nStep: 2\n\nNeuron 1\n0|1 \n");
}


TEST(SpikesLogger, ModelMonitoring)
{
    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};

    {  //stop spikes from happening
        const auto& params = population.get_neurons_parameters();
        for (size_t i = 0; i < params.size(); i++)
        {
            auto param = params[i];
            param.activation_threshold_ = std::numeric_limits<double>::max();
            //todo change this to set_neuron_parameters when #69 gets fixed
            population.set_neurons_parameters(i, param);
        }
    }

    knp::testing::DeltaProjection input_projection = knp::testing::DeltaProjection{
        knp::core::UID{false}, population.get_uid(), knp::testing::input_projection_gen, 1};
    knp::framework::Network network;
    network.add_population(std::move(population));
    network.add_projection<knp::testing::DeltaProjection>(std::move(input_projection));

    const knp::core::UID i_channel_uid, o_channel_uid;

    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_projection.get_uid());
    model.add_output_channel(o_channel_uid, population.get_uid());

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(
        model, backend_loader.load(knp::testing::get_backend_path()),
        {{i_channel_uid,
          [](knp::core::Step step) -> knp::core::messaging::SpikeData
          {
              if (step % 2 == 0)
              {
                  knp::core::messaging::SpikeData spike_data;
                  spike_data.push_back(0);
                  return spike_data;
              }
              return {};
          }}});

    std::ostringstream projection_weights_stream;

    knp::framework::monitoring::model_monitoring::add_spikes_logger(
        model, {{i_channel_uid, "INPUT"}}, model_executor, projection_weights_stream);
    model_executor.start([](size_t step) -> bool { return step < 3; });

    ASSERT_STREQ(projection_weights_stream.str().c_str(), "Step: 0\nSender: INPUT\n0 \nStep: 2\nSender: INPUT\n0 \n");
}
