/**
 * @file spike_generators.cpp
 * @brief Functions for creating specific spikes generators.
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

#include "spike_generators.h"

#include <settings.h>


template <>
std::function<knp::core::messaging::SpikeData(knp::core::Step)>
make_training_images_spikes_generator<knp::neuron_traits::AltAILIF>(const Dataset& dataset)
{
    return dataset.make_training_images_spikes_generator();
}


template <>
std::function<knp::core::messaging::SpikeData(knp::core::Step)>
make_training_labels_spikes_generator<knp::neuron_traits::AltAILIF>(const Dataset& dataset)
{
    return [&dataset](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;

        knp::core::Step local_step = step % steps_per_image;
        if (local_step == 11) message.push_back(dataset.get_data_for_training().first[step / steps_per_image].first);
        return message;
    };
}


template <>
std::function<knp::core::messaging::SpikeData(knp::core::Step)>
make_inference_images_spikes_generator<knp::neuron_traits::AltAILIF>(const Dataset& dataset)
{
    return dataset.make_inference_images_spikes_generator();
}
