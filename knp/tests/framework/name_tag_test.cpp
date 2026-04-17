/**
 * @file name_tag_test.cpp
 * @brief Tests for name tag.
 * @kaspersky_support David P.
 * @date 06.04.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <knp/framework/population/neuron_parameters_generators.h>
#include <knp/framework/tags/name.h>

#include <tests_common.h>


TEST(NameTag, DefaultName)
{
    knp::core::UID uid;
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> pop(
        uid, knp::framework::population::neurons_generators::make_default<knp::neuron_traits::BLIFATNeuron>(), 0);
    ASSERT_EQ(knp::framework::tags::get_name(pop), std::string(uid));
}


TEST(NameTag, SetGetName)
{
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> pop(
        knp::framework::population::neurons_generators::make_default<knp::neuron_traits::BLIFATNeuron>(), 0);
    knp::framework::tags::set_name(pop, "test_name");
    ASSERT_EQ(knp::framework::tags::get_name(pop), "test_name");
}
