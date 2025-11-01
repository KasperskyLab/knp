/**
 * @file construct_network.cpp
 * @brief Functions for network construction.
 * @kaspersky_support A. Vartenkov
 * @date 03.12.2024
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

#include "construct_network.h"

#include "shared_network.h"


// A list of short type names to make reading easier.
using DeltaSynapseParams = knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using DeltaSynapse = knp::synapse_traits::DeltaSynapse;
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using ResourceSynapseData = ResourceDeltaProjection::Synapse;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
using AltAILIFPopulation = knp::core::Population<knp::neuron_traits::AltAILIF>;
using ResourceAltAILIFPopulation = knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron>;
using ResourceNeuron = knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron;
using ResourceNeuronData = knp::neuron_traits::neuron_parameters<ResourceNeuron>;


// Intermediate population neurons.
template <class Neuron>
struct PopulationData
{
    size_t size_;
    knp::neuron_traits::neuron_parameters<Neuron> neuron_;
};


enum PopIndexes
{
    L,
    WTA,
    REWGATE,
    OUT,
    BIASGATE
};


// Calculate synaptic resource value given synapse weight.
float resource_from_weight(float weight, float min_weight, float max_weight)
{
    // Max weight is only possible with infinite resource, so we should select a value less than that.
    float eps = 1e-6;
    if (min_weight > max_weight) std::swap(min_weight, max_weight);
    if (weight < min_weight || weight >= max_weight - eps)
        throw std::logic_error("Weight should not be less than min_weight, more than max_weight or too close to it.");
    double diff = max_weight - min_weight;
    double over = weight - min_weight;
    return static_cast<float>(over * diff / (diff - over));
}

// ARNI activation threshold constant
constexpr uint16_t threshold_idk_constant = 8531;

// Add populations to the network.
auto add_subnetwork_populations(AnnotatedNetwork &result)
{
    ResourceNeuronData default_neuron;
    default_neuron.activation_threshold_ = threshold_idk_constant;
    ResourceNeuronData l_neuron = default_neuron;
    l_neuron.potential_leak_ = -l_neuron.activation_threshold_ / 6;
    l_neuron.negative_activation_threshold_ = 0;
    l_neuron.potential_reset_value_ = 0;

    l_neuron.dopamine_plasticity_time_ = 10;
    l_neuron.isi_max_ = 10;
    l_neuron.d_h_ = -0.277539;

    // stability_resource_change_ratio
    l_neuron.stability_change_parameter_ = 24.5291f;

    // silent synapses
    l_neuron.resource_drain_coefficient_ = 2;

    // chartime * hebbian_plasticity_chartime_ratio
    l_neuron.dopamine_plasticity_time_ = 6 * 2.72f;

    // threshold_excess_weight_dependent
    l_neuron.synapse_sum_threshold_coefficient_ = 0.0274326f;

    struct PopulationRole
    {
        PopulationData<ResourceNeuron> pd_;
        bool for_inference_;
        bool output_;
        std::string name_;
    };
    //
    std::vector<PopulationRole> pop_data{
        {{30, l_neuron}, true, false, "L"},
        {{30, default_neuron}, true, false, "WTA"},
        {{30, default_neuron}, true, false, "REWGATE"},
        {{10, default_neuron}, true, true, "OUT"},
        {{10, default_neuron}, true, false, "BIASGATE"}};

    std::vector<knp::core::UID> population_uids;
    for (auto &pop_init_data : pop_data)
    {
        // A very simple neuron generator returning a default neuron.
        auto neuron_generator = [&pop_init_data](size_t index) { return pop_init_data.pd_.neuron_; };

        knp::core::UID uid;
        result.network_.add_population(ResourceAltAILIFPopulation(uid, neuron_generator, pop_init_data.pd_.size_));
        population_uids.push_back(uid);
        result.data_.population_names_[uid] = pop_init_data.name_;
        if (pop_init_data.for_inference_) result.data_.inference_population_uids_.insert(uid);
        if (pop_init_data.output_) result.data_.output_uids_.push_back(uid);
    }

    result.data_.wta_data_.emplace_back().first.push_back(population_uids[WTA]);
    return std::make_pair(population_uids, pop_data);
}

// Create network for MNIST.
AnnotatedNetwork create_example_network(int num_compound_networks)
{
    AnnotatedNetwork result;
    for (int i = 0; i < num_compound_networks; ++i)
    {
        auto [population_uids, pop_data] = add_subnetwork_populations(result);

        /*

        do we really need this?

        // Now that we added all the populations we need, we have to connect them with projections.
        // Creating a projection is more tricky, as all the connection logic should be described in a generator.
        // Create a default synapse.
        ResourceSynapseParams default_synapse;
        auto afferent_synapse = default_synapse;
        afferent_synapse.rule_.synaptic_resource_ =
            resource_from_weight(base_weight_value, min_synaptic_weight, max_synaptic_weight);
        afferent_synapse.rule_.dopamine_plasticity_period_ = synapse_dopamine_period;
        afferent_synapse.rule_.w_min_ = min_synaptic_weight;
        afferent_synapse.rule_.w_max_ = max_synaptic_weight;
        */


        ResourceSynapseParams R_to_L_synapse;
        //-weight_inc
        // R_to_L_synapse.rule_.d_u_ = 0.277539;
        // w_min and w_max * 1000 because of 8531
        R_to_L_synapse.rule_.dopamine_plasticity_period_ = 10;
        R_to_L_synapse.rule_.w_min_ = -1.02827 * 1000.f;
        // R_to_L_synapse.rule_.synaptic_resource_ = 4.263f * 1000.f;

        //R_to_L_synapse.rule_.w_max_ = 320.f;
        //R_to_L_synapse.rule_.synaptic_resource_ =
        //    resource_from_weight(0, R_to_L_synapse.rule_.w_min_, R_to_L_synapse.rule_.w_max_);


        std::mt19937 rand_engine((std::random_device())());
        std::uniform_real_distribution<float> wmax_distribution(0.282071f, 0.371639f);

        ResourceDeltaProjection R_to_L_projection = knp::framework::projection::creators::all_to_all<ResourceSynapse>(
            knp::core::UID(false), population_uids[L], input_size, pop_data[L].pd_.size_,
            [&R_to_L_synapse, &rand_engine, &wmax_distribution](size_t i1, size_t i2)
            {
                // if (i2 % 1 == 0 )//&& i1 == 0)
                //{
                R_to_L_synapse.rule_.w_max_ = wmax_distribution(rand_engine) * 1000.f;
                R_to_L_synapse.rule_.synaptic_resource_ =
                    resource_from_weight(0, R_to_L_synapse.rule_.w_min_, R_to_L_synapse.rule_.w_max_);
                //}

                return R_to_L_synapse;
            });
        result.data_.projections_from_raster_.push_back(R_to_L_projection.get_uid());
        R_to_L_projection.unlock_weights();  // Trainable
        result.network_.add_projection(R_to_L_projection);
        result.data_.inference_internal_projection_.insert(R_to_L_projection.get_uid());

        DeltaSynapseParams L_to_WTA_synapse;
        L_to_WTA_synapse.weight_ = 20.f * 1000.f;

        DeltaProjection L_to_WTA_projection = knp::framework::projection::creators::aligned<DeltaSynapse>(
            population_uids[L], population_uids[WTA], pop_data[L].pd_.size_, pop_data[WTA].pd_.size_,
            [&L_to_WTA_synapse](size_t, size_t) { return L_to_WTA_synapse; });
        result.network_.add_projection(L_to_WTA_projection);
        result.data_.inference_internal_projection_.insert(L_to_WTA_projection.get_uid());


        DeltaSynapseParams WTA_to_REWGATE_synapse;
        WTA_to_REWGATE_synapse.weight_ = 10.f * 1000.f;
        WTA_to_REWGATE_synapse.delay_ = 2;
        WTA_to_REWGATE_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;

        DeltaProjection WTA_to_REWGATE_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[WTA], population_uids[REWGATE], pop_data[WTA].pd_.size_, pop_data[REWGATE].pd_.size_,
                [&WTA_to_REWGATE_synapse](size_t, size_t) { return WTA_to_REWGATE_synapse; });
        result.network_.add_projection(WTA_to_REWGATE_projection);
        result.data_.inference_internal_projection_.insert(WTA_to_REWGATE_projection.get_uid());
        result.data_.wta_data_[i].second.push_back(WTA_to_REWGATE_projection.get_uid());


        DeltaSynapseParams REWGATE_to_L_synapse;
        REWGATE_to_L_synapse.output_type_ = knp::synapse_traits::OutputType::DOPAMINE;
        REWGATE_to_L_synapse.weight_ = 0.835402f;


        // lowk i dont understand one thing, in michail's system synapse is a part of neuron,
        // so its parameters should distribute to other synapses going to this neuron or no?
        // also if synapse is a part of neuron then how there can be synaptic and not synaptic
        // synapses connected at same time? something aint right...
        // to be fair it is important for all synapses, not only about L, need to remember ts.

        DeltaProjection REWGATE_to_L_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[REWGATE], population_uids[L], pop_data[REWGATE].pd_.size_, pop_data[L].pd_.size_,
                [&REWGATE_to_L_synapse](size_t, size_t) { return REWGATE_to_L_synapse; });
        result.network_.add_projection(REWGATE_to_L_projection);
        result.data_.inference_internal_projection_.insert(REWGATE_to_L_projection.get_uid());


        DeltaSynapseParams WTA_to_OUT_synapse;
        WTA_to_OUT_synapse.weight_ = 20.f * 1000.f;

        DeltaProjection WTA_to_OUT_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[WTA], population_uids[OUT], pop_data[WTA].pd_.size_, pop_data[OUT].pd_.size_,
                [&WTA_to_OUT_synapse](size_t, size_t) { return WTA_to_OUT_synapse; });
        result.network_.add_projection(WTA_to_OUT_projection);
        result.data_.inference_internal_projection_.insert(WTA_to_OUT_projection.get_uid());
        result.data_.wta_data_[i].second.push_back(WTA_to_OUT_projection.get_uid());


        DeltaSynapseParams OUT_to_BIASGATE_synapse;
        OUT_to_BIASGATE_synapse.weight_ = -10.f;
        OUT_to_BIASGATE_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;

        DeltaProjection OUT_to_BIASGATE_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[OUT], population_uids[BIASGATE], pop_data[OUT].pd_.size_, pop_data[BIASGATE].pd_.size_,
                [&OUT_to_BIASGATE_synapse](size_t, size_t) { return OUT_to_BIASGATE_synapse; });
        result.network_.add_projection(OUT_to_BIASGATE_projection);
        result.data_.inference_internal_projection_.insert(OUT_to_BIASGATE_projection.get_uid());


        DeltaSynapseParams TARGET_to_REWGATE_synapse;
        TARGET_to_REWGATE_synapse.weight_ = 20.f * 1000.f;

        DeltaProjection TARGET_to_REWGATE_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                knp::core::UID(false), population_uids[REWGATE], 10, pop_data[REWGATE].pd_.size_,
                [&TARGET_to_REWGATE_synapse](size_t, size_t) { return TARGET_to_REWGATE_synapse; });
        result.data_.projections_from_classes_.push_back(TARGET_to_REWGATE_projection.get_uid());
        result.network_.add_projection(TARGET_to_REWGATE_projection);
        result.data_.inference_internal_projection_.insert(TARGET_to_REWGATE_projection.get_uid());


        DeltaSynapseParams BIASGATE_to_REWGATE_synapse;
        BIASGATE_to_REWGATE_synapse.weight_ = 20.f * 1000.f;

        DeltaProjection BIASGATE_to_REWGATE_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[BIASGATE], population_uids[REWGATE], pop_data[BIASGATE].pd_.size_,
                pop_data[REWGATE].pd_.size_,
                [&BIASGATE_to_REWGATE_synapse](size_t, size_t) { return BIASGATE_to_REWGATE_synapse; });
        result.network_.add_projection(BIASGATE_to_REWGATE_projection);
        result.data_.inference_internal_projection_.insert(BIASGATE_to_REWGATE_projection.get_uid());


        DeltaSynapseParams TARGET_to_BIASGATE_synapse;
        TARGET_to_BIASGATE_synapse.weight_ = 20.f * 1000.f;

        DeltaProjection TARGET_to_BIASGATE_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                knp::core::UID(false), population_uids[BIASGATE], 10, pop_data[BIASGATE].pd_.size_,
                [&TARGET_to_BIASGATE_synapse](size_t, size_t) { return TARGET_to_BIASGATE_synapse; });
        result.data_.projections_from_classes_.push_back(TARGET_to_BIASGATE_projection.get_uid());
        result.network_.add_projection(TARGET_to_BIASGATE_projection);
        result.data_.inference_internal_projection_.insert(TARGET_to_BIASGATE_projection.get_uid());


        DeltaSynapseParams BIASGATE_to_L_synapse;
        BIASGATE_to_L_synapse.weight_ = 3.f * 1000.f;

        DeltaProjection BIASGATE_to_L_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                population_uids[BIASGATE], population_uids[L], pop_data[BIASGATE].pd_.size_, pop_data[L].pd_.size_,
                [&BIASGATE_to_L_synapse](size_t, size_t) { return BIASGATE_to_L_synapse; });
        result.network_.add_projection(BIASGATE_to_L_projection);
        result.data_.inference_internal_projection_.insert(BIASGATE_to_L_projection.get_uid());


        DeltaSynapseParams TARGET_to_L_synapse;
        TARGET_to_L_synapse.weight_ = -1.f;
        TARGET_to_L_synapse.delay_ = 11;
        TARGET_to_L_synapse.output_type_ = knp::synapse_traits::OutputType::BLOCKING;

        DeltaProjection TARGET_to_L_projection =
            knp::framework::projection::creators::aligned<knp::synapse_traits::DeltaSynapse>(
                knp::core::UID(false), population_uids[L], 10, pop_data[L].pd_.size_,
                [&TARGET_to_L_synapse](size_t, size_t) { return TARGET_to_L_synapse; });
        result.data_.projections_from_classes_.push_back(TARGET_to_L_projection.get_uid());
        result.network_.add_projection(TARGET_to_L_projection);
        result.data_.inference_internal_projection_.insert(TARGET_to_L_projection.get_uid());
    }

    // Return created network.
    return result;
}
