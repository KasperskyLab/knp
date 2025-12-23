/**
 * @file message_handlers.cpp
 * @brief Implementation of message handler functionality.
 * @kaspersky_support  A. Vartenkov
 * @date 25.11.2024
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

#include <knp/framework/message_handlers.h>

#include <unordered_set>
#include <utility>


/**
 * @brief namespace for message modifier callables.
 */
namespace knp::framework::modifier
{

knp::core::messaging::SpikeData select_random_n(
    knp::core::messaging::SpikeData &input, size_t n, std::mt19937 &random_engine)
{
    if (input.size() <= n) return input;
    std::uniform_int_distribution<size_t> distribution;
    knp::core::messaging::SpikeData result;
    for (size_t i = 0; i < n; ++i)
    {
        const size_t index = distribution(random_engine) % (input.size() - i);
        result.push_back(input[index]);
        std::swap(input[index], input[input.size() - 1 - i]);
    }
    return result;
}


knp::core::messaging::SpikeData KWtaRandomHandler::operator()(std::vector<knp::core::messaging::SpikeMessage> &messages)
{
    if (messages.empty())
    {
        return {};
    }

    auto &msg = messages[0];
    if (msg.neuron_indexes_.size() < num_winners_)
    {
        return msg.neuron_indexes_;
    }

    knp::core::messaging::SpikeData out_spikes = select_random_n(msg.neuron_indexes_, num_winners_, random_engine_);

    return out_spikes;
}


knp::core::messaging::SpikeData GroupWtaRandomHandler::operator()(
    const std::vector<knp::core::messaging::SpikeMessage> &messages)
{
    if (messages.empty())
    {
        return {};
    }

    if (num_winners_ > group_borders_.size())
    {
        return messages[0].neuron_indexes_;
    }

    const auto &spikes = messages[0].neuron_indexes_;
    if (spikes.empty())
    {
        return {};
    }

    std::vector<knp::core::messaging::SpikeData> spikes_per_group(group_borders_.size() + 1);

    // Fill groups in.
    for (const auto &spike : spikes)
    {
        const size_t group_index =
            std::upper_bound(group_borders_.begin(), group_borders_.end(), spike) - group_borders_.begin();
        spikes_per_group[group_index].push_back(spike);
    }

    // Sort groups by number of elements.
    std::sort(
        spikes_per_group.begin(), spikes_per_group.end(),
        [](const auto &el1, const auto &el2) { return el1.size() > el2.size(); });

    // Find all groups with the same number of spikes as the K-th one.
    const auto &last_group = spikes_per_group[num_winners_ - 1];
    auto group_interval = std::equal_range(
        spikes_per_group.begin(), spikes_per_group.end(), last_group,
        [](const auto &el1, const auto &el2) { return el1.size() > el2.size(); });
    
    assert(group_interval.first - spikes_per_group.begin() + 1 >= 0);
    assert(static_cast<size_t>(group_interval.first - spikes_per_group.begin() + 1) <= num_winners_);
    
    // The approach could be more efficient, but I don't think it's necessary.
    std::shuffle(group_interval.first, group_interval.second, random_engine_);
    knp::core::messaging::SpikeData result;
    for (size_t i = 0; i < num_winners_; ++i)
    {
        for (const auto &spike : spikes_per_group[i])
        {
            result.push_back(spike);
        }
    }
    return result;
}


knp::core::messaging::SpikeData KWtaPerGroup::operator()(
    const std::vector<knp::core::messaging::SpikeMessage> &messages)
{
    if (messages.empty()) return {};

    auto spikes = messages[0].neuron_indexes_;
    if (spikes.empty()) return {};

    std::vector<knp::core::messaging::SpikeData> spikes_per_group(group_borders_.size() + 1);
    for (const auto &spike : spikes)
    {
        const size_t group_index =
            std::upper_bound(group_borders_.begin(), group_borders_.end(), spike) - group_borders_.begin();
        spikes_per_group[group_index].push_back(spike);
    }

    knp::core::messaging::SpikeData result;
    result.reserve(group_borders_.size() * winners_per_group_);
    for (auto &spike_group : spikes_per_group)
    {
        knp::core::messaging::SpikeData result_buf = select_random_n(spike_group, winners_per_group_, random_engine_);
        result.insert(result.end(), result_buf.begin(), result_buf.end());
    }
    return result;
}


knp::core::messaging::SpikeData SpikeUnionHandler::operator()(
    const std::vector<knp::core::messaging::SpikeMessage> &messages)
{
    std::unordered_set<knp::core::messaging::SpikeIndex> spikes;
    for (const auto &msg : messages)
    {
        spikes.insert(msg.neuron_indexes_.begin(), msg.neuron_indexes_.end());
    }
    knp::core::messaging::SpikeData result;
    result.reserve(spikes.size());
    std::copy(spikes.begin(), spikes.end(), std::back_inserter(result));
    return result;
}
}  // namespace knp::framework::modifier
