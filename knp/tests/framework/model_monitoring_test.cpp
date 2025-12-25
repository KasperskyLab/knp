/**
 * @file model_monitoring_test.cpp
 * @brief Model monitoring test.
 * @kaspersky_support D. Postnikov
 * @date 10.08.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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
#include <knp/framework/monitoring/model.h>

#include <generators.h>
#include <tests_common.h>


TEST(ModelMonitoring, AggregatedSpikesLogger)
{
    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};

    {  //stop spikes from happening
        auto params = population.get_neurons_parameters();
        for (auto& param : params)
        {
            param.activation_threshold_ = std::numeric_limits<double>::max();
        }
        population.set_neurons_parameters(params);
    }

    knp::testing::DeltaProjection input_projection = knp::testing::DeltaProjection{
        knp::core::UID{false}, population.get_uid(), knp::testing::input_projection_gen, 1};

    knp::framework::Network network;

    const knp::core::UID input_projection_uid = input_projection.get_uid();
    const knp::core::UID population_uid = population.get_uid();

    network.add_population(std::move(population));
    network.add_projection<knp::testing::DeltaProjection>(std::move(input_projection));

    const knp::core::UID i_channel_uid, o_channel_uid;

    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_projection_uid);
    model.add_output_channel(o_channel_uid, population_uid);

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

    std::map<std::string, size_t> spike_accumulator;
    std::ostringstream projection_weights_stream;

    knp::framework::monitoring::model::add_aggregated_spikes_logger(
        model, {{i_channel_uid, "INPUT"}}, model_executor, spike_accumulator, projection_weights_stream, 1);
    model_executor.start([](size_t step) -> bool { return step < 3; });

    ASSERT_EQ(projection_weights_stream.str(), "Index, INPUT\n1, 0\n2, 1\n3, 0\n");
}


TEST(ModelMonitoring, ProjectionWeightsLogger)
{
    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};

    knp::testing::ResourceSynapseParams default_synapse;

    knp::testing::ResourceDeltaProjection input_projection{
        knp::core::UID{false}, population.get_uid(),
        [&](size_t) {
            return knp::testing::ResourceSynapseData{default_synapse, 0, 0};
        },
        1};

    knp::framework::Network network;

    const knp::core::UID input_projection_uid = input_projection.get_uid();
    const knp::core::UID population_uid = population.get_uid();

    network.add_population(std::move(population));
    network.add_projection<knp::testing::ResourceDeltaProjection>(std::move(input_projection));

    const knp::core::UID i_channel_uid, o_channel_uid;

    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_projection_uid);
    model.add_output_channel(o_channel_uid, population_uid);

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
    knp::framework::monitoring::model::add_projection_weights_logger(
        projection_weights_stream, model_executor, input_projection.get_uid(), 1);
    model_executor.start([](size_t step) -> bool { return step < 2; });

    ASSERT_EQ(projection_weights_stream.str(), "Step: 1\n\nNeuron 0\n0|0 \nStep: 2\n\nNeuron 0\n0|1 \n");
}


TEST(ModelMonitoring, SpikesLogger)
{
    knp::testing::BLIFATPopulation population{knp::testing::neuron_generator, 1};

    {  //stop spikes from happening
        auto params = population.get_neurons_parameters();
        for (auto& param : params)
        {
            param.activation_threshold_ = std::numeric_limits<double>::max();
        }
        population.set_neurons_parameters(params);
    }

    knp::testing::DeltaProjection input_projection = knp::testing::DeltaProjection{
        knp::core::UID{false}, population.get_uid(), knp::testing::input_projection_gen, 1};

    knp::framework::Network network;

    const knp::core::UID input_projection_uid = input_projection.get_uid();
    const knp::core::UID population_uid = population.get_uid();

    network.add_population(std::move(population));
    network.add_projection<knp::testing::DeltaProjection>(std::move(input_projection));

    const knp::core::UID i_channel_uid, o_channel_uid;

    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_projection_uid);
    model.add_output_channel(o_channel_uid, population_uid);

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

    knp::framework::monitoring::model::add_spikes_logger(
        model_executor, {{i_channel_uid, "INPUT"}}, projection_weights_stream);
    model_executor.start([](size_t step) -> bool { return step < 3; });

    ASSERT_EQ(projection_weights_stream.str(), "Step: 0\nSender: INPUT\n0 \nStep: 2\nSender: INPUT\n0 \n");
}
