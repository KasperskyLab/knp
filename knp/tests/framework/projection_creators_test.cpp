/**
 * @file projection_creators_test.cpp
 * @brief Tests for projection creators.
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

#include <knp/framework/projection/creators.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>

#include <vector>


TEST(ProjectionConnectors, AllToAll)
{
    constexpr size_t src_pop_size = 2;
    constexpr size_t dest_pop_size = 4;

    auto proj = knp::framework::projection::creators::all_to_all<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), src_pop_size, dest_pop_size);

    ASSERT_EQ(proj.size(), src_pop_size * dest_pop_size);

    size_t index = 0;
    for (const auto& synapse : proj)
    {
        const auto source_syn_index = std::get<knp::core::source_neuron_id>(synapse);
        const auto target_syn_index = std::get<knp::core::target_neuron_id>(synapse);

        SPDLOG_DEBUG("Synapse: {} -> {}.", source_syn_index, target_syn_index);
        ASSERT_EQ(source_syn_index, index % src_pop_size);
        ASSERT_EQ(target_syn_index, index / src_pop_size);
        ++index;
    }
}


TEST(ProjectionConnectors, Aligned)
{
    constexpr size_t src_pop_size = 3;
    constexpr size_t dest_pop_size = 6;

    auto proj = knp::framework::projection::creators::aligned<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), src_pop_size, dest_pop_size);

    ASSERT_EQ(proj.size(), std::max(src_pop_size, dest_pop_size));

    size_t index = 0;
    for (const auto& synapse : proj)
    {
        const auto source_syn_index = std::get<knp::core::source_neuron_id>(synapse);
        const auto target_syn_index = std::get<knp::core::target_neuron_id>(synapse);

        SPDLOG_DEBUG("Synapse: {} -> {}.", source_syn_index, target_syn_index);
        ASSERT_EQ(target_syn_index, index);
        ASSERT_EQ(source_syn_index, index / 2);
        ++index;
    }
}


TEST(ProjectionConnectors, Exclusive)
{
    constexpr size_t pops_size = 3;

    auto proj = knp::framework::projection::creators::exclusive<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), pops_size);

    ASSERT_EQ(proj.size(), pops_size * (pops_size - 1));

    constexpr std::array<size_t, 6> correct_target{1, 2, 0, 2, 0, 1};

    size_t index = 0;
    for (const auto& synapse : proj)
    {
        const auto source_syn_index = std::get<knp::core::source_neuron_id>(synapse);
        const auto target_syn_index = std::get<knp::core::target_neuron_id>(synapse);

        SPDLOG_DEBUG("Synapse: {} -> {}.", source_syn_index, target_syn_index);
        ASSERT_EQ(source_syn_index, index / 2);
        ASSERT_EQ(target_syn_index, correct_target[index]);
        ++index;
    }
}


TEST(ProjectionConnectors, OneToOne)
{
    constexpr size_t pop_size = 5;

    auto proj = knp::framework::projection::creators::one_to_one<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), pop_size);

    ASSERT_EQ(proj.size(), pop_size);

    for (const auto& synapse : proj)
    {
        const auto source_syn_index = std::get<knp::core::source_neuron_id>(synapse);
        const auto target_syn_index = std::get<knp::core::target_neuron_id>(synapse);

        SPDLOG_DEBUG("Synapse: {} -> {}.", source_syn_index, target_syn_index);
        ASSERT_EQ(source_syn_index, target_syn_index);
    }
}


TEST(ProjectionConnectors, FromContainer)
{
    constexpr auto e_count = 5;
    std::vector<typename knp::core::Projection<knp::synapse_traits::DeltaSynapse>::Synapse> container;
    container.reserve(e_count);

    for (int i = 0; i < e_count; ++i)
    {
        container.push_back(std::make_tuple(
            knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters(), i, e_count - i));
    }

    auto proj =
        knp::framework::projection::creators::from_container<typename knp::synapse_traits::DeltaSynapse, std::vector>(
            knp::core::UID(), knp::core::UID(), container);

    int syn_num = 0;
    for (const auto& synapse : proj)
    {
        const auto container_syn = container[syn_num++];
        ASSERT_EQ(std::get<knp::core::target_neuron_id>(synapse), std::get<knp::core::target_neuron_id>(container_syn));
        ASSERT_EQ(std::get<knp::core::source_neuron_id>(synapse), std::get<knp::core::source_neuron_id>(container_syn));
    }
}


TEST(ProjectionConnectors, FromMap)
{
    const auto e_count = 5;
    std::map<
        typename std::tuple<size_t, size_t>,
        typename knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters>
        syn_map;

    for (int i = 0; i < e_count; ++i)
    {
        syn_map[std::make_tuple(i, e_count - i)] =
            knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters();
    }

    auto proj = knp::framework::projection::creators::from_map<typename knp::synapse_traits::DeltaSynapse, std::map>(
        knp::core::UID(), knp::core::UID(), syn_map);

    for (const auto& synapse : proj)
    {
        // Find in map.
        ASSERT_NE(
            syn_map.find(std::make_tuple(
                std::get<knp::core::source_neuron_id>(synapse), std::get<knp::core::target_neuron_id>(synapse))),
            syn_map.end());
    }
}


TEST(ProjectionConnectors, FixedProbability)
{
    [[maybe_unused]] auto proj =  //!OCLINT
        knp::framework::projection::creators::fixed_probability<typename knp::synapse_traits::DeltaSynapse>(
            knp::core::UID(), knp::core::UID(), 3, 5, 0.5);
}


TEST(ProjectionConnectors, IndexBased)
{
    constexpr size_t src_pop_size = 5;
    constexpr size_t dest_pop_size = 3;

    auto proj = knp::framework::projection::creators::index_based<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), src_pop_size, dest_pop_size,
        [dest_pop_size](size_t index0, size_t index1)
            -> std::optional<typename knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters>
        {
            // Diagonal.
            if (index0 == index1)
                return std::make_optional<
                    typename knp::core::Projection<knp::synapse_traits::DeltaSynapse>::SynapseParameters>();
            return std::nullopt;
        });

    ASSERT_EQ(proj.size(), dest_pop_size);
}


TEST(ProjectionConnectors, FixedNumberPost)
{
    constexpr size_t src_pop_size = 3;
    constexpr size_t dest_pop_size = 5;
    constexpr size_t conn_count = 3;

    auto proj = knp::framework::projection::creators::fixed_number_post<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), src_pop_size, dest_pop_size, conn_count);

    ASSERT_EQ(proj.size(), src_pop_size * conn_count);

    std::map<int, int> index_map;

    for (const auto& synapse : proj)
    {
        const auto target_syn_index = std::get<knp::core::target_neuron_id>(synapse);
        const auto source_syn_index = std::get<knp::core::source_neuron_id>(synapse);

        SPDLOG_DEBUG("Synapse: {} -> {}.", target_syn_index, source_syn_index);

        ++index_map[source_syn_index];
    }

    for (const auto& [key, value] : index_map)
    {
        ASSERT_EQ(value, conn_count);
    }
}


TEST(ProjectionConnectors, FixedNumberPre)
{
    constexpr size_t src_pop_size = 4;
    constexpr size_t dest_pop_size = 8;
    constexpr size_t conn_count = 3;

    auto proj = knp::framework::projection::creators::fixed_number_pre<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), src_pop_size, dest_pop_size, conn_count);

    ASSERT_EQ(proj.size(), dest_pop_size * conn_count);

    std::map<int, int> index_map;

    for (const auto& synapse : proj)
    {
        const auto target_syn_index = std::get<knp::core::target_neuron_id>(synapse);
        const auto source_syn_index = std::get<knp::core::source_neuron_id>(synapse);

        SPDLOG_DEBUG("Synapse: {} -> {}.", target_syn_index, source_syn_index);

        ++index_map[target_syn_index];
    }

    for (const auto& [key, value] : index_map)
    {
        ASSERT_EQ(value, conn_count);
    }
}


TEST(ProjectionConnectors, CloneProjection)
{
    constexpr size_t pop_size = 3;

    auto proj = knp::framework::projection::creators::one_to_one<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), pop_size);

    auto new_proj = knp::framework::projection::creators::clone_projection<typename knp::synapse_traits::DeltaSynapse>(
        proj, [&proj](size_t index) { return std::get<knp::core::synapse_data>(proj[index]); });

    ASSERT_EQ(new_proj.size(), proj.size());

    for (int i = 0; i < proj.size(); ++i)
    {
        ASSERT_EQ(std::get<knp::core::target_neuron_id>(proj[i]), std::get<knp::core::target_neuron_id>(new_proj[i]));
        ASSERT_EQ(std::get<knp::core::source_neuron_id>(proj[i]), std::get<knp::core::source_neuron_id>(new_proj[i]));
    }
}
