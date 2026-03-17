/**
 * @file projection_synapse_parameters_generators_test.cpp
 * @brief Tests for projection synapse parameters generators.
 * @kaspersky_support Artiom N.
 * @date 27.08.2024
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


#include <knp/core/projection.h>
#include <knp/framework/projection/synapse_parameters_generators.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>

#include <vector>


TEST(ProjectionSynapsesParametersGenerators, Default1Param)
{
    const typename knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters def_syn;
    const auto new_syn = knp::framework::projection::parameters_generators::default_synapse_gen_1param<
        knp::synapse_traits::DeltaSynapse>(0);

    ASSERT_EQ(def_syn.weight_, new_syn.weight_);
    ASSERT_EQ(def_syn.delay_, new_syn.delay_);
    ASSERT_EQ(def_syn.output_type_, new_syn.output_type_);
}


TEST(ProjectionSynapsesParametersGenerators, Default2Param)
{
    const typename knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters def_syn;
    const auto new_syn = knp::framework::projection::parameters_generators::default_synapse_gen_2param<
        knp::synapse_traits::DeltaSynapse>(0, 0);

    ASSERT_EQ(def_syn.weight_, new_syn.weight_);
    ASSERT_EQ(def_syn.delay_, new_syn.delay_);
    ASSERT_EQ(def_syn.output_type_, new_syn.output_type_);
}


TEST(ProjectionSynapsesParametersGenerators, Copy)
{
    const typename knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters def_syn;
    const typename knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters src_syn{
        123, 321, knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE};

    const auto syn_gen =
        knp::framework::projection::parameters_generators::CopySynapseGen<knp::synapse_traits::DeltaSynapse>{src_syn};
    const auto new_syn = syn_gen(0, 0);

    ASSERT_NE(def_syn.weight_, new_syn.weight_);
    ASSERT_NE(def_syn.delay_, new_syn.delay_);
    ASSERT_NE(def_syn.output_type_, new_syn.output_type_);

    ASSERT_EQ(src_syn.weight_, new_syn.weight_);
    ASSERT_EQ(src_syn.delay_, new_syn.delay_);
    ASSERT_EQ(src_syn.output_type_, new_syn.output_type_);
}
