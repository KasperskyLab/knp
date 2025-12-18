/**
 * @file message_handlers.h
 * @brief A set of predefined message handling functions to add to model executor.
 * @kaspersky_support Vartenkov A.
 * @date 19.11.2024
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
#pragma once

#include <knp/core/impexp.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>


/**
 * @brief Modifier namespace.
 */
namespace knp::framework::modifier
{

/**
 * @brief The KWtaRandomHandler class is a definition of a message handler functor that processes 
 * spike messages and selects random N spikes out of the whole set.
 * @note The modifier processes only one message per step.
 */
class KNP_DECLSPEC KWtaRandomHandler
{
public:
    /**
     * @brief Functor constructor.
     * @param winners_number maximum number of groups to pass spikes further.
     * @param seed random generator seed.
     * @note The constructor uses `mt19937` generator algorithm for random number generation.
     */
    explicit KWtaRandomHandler(size_t winners_number = 1, int seed = 0)
        : num_winners_(winners_number), random_engine_(seed)
    {
    }

    /**
     * @brief Call operator to process a number of messages and return indexes of spiked neurons.
     * @param messages vector of spike messages.
     * @return random indexes of no more than N spiked neurons.
     * @note It is assumed that the method receives no more than one message per step. 
     * Therefore, all messages except the first one in the `messages` parameter are ignored.
     */
    knp::core::messaging::SpikeData operator()(std::vector<knp::core::messaging::SpikeMessage> &messages);

private:
    size_t num_winners_;
    std::mt19937 random_engine_;
};


/**
 * @brief The GroupWtaRandomHandler class is a definition of a message handler functor that 
 * passes spikes further from no more than a fixed number of groups at once.
 * @details The functor divides spike messages into groups, then sorts the groups in the 
 * descending order based on the number of spikes.
 * @note If the last place in the top N is shared between groups, the functor selects randomly 
 * among these groups.
 */
class KNP_DECLSPEC GroupWtaRandomHandler
{
public:
    /**
     * @brief Functor constructor.
     * @note For example, we have a set of spike messages 0, 1, 2, 3, 4, 5. If `group_borders` 
     * is {2, 4}, the set of spike messages will be divided into the following groups: 
     * [0, 1], [2, 3], and [4, 5].
     * @param group_borders vector of spike message indexes that define right borders for 
     * each group.
     * @param num_winning_groups maximum number of groups that can pass their spikes further.
     * @param seed random generator seed.
     */
    explicit GroupWtaRandomHandler(
        const std::vector<size_t> &group_borders, size_t num_winning_groups = 1, int seed = 0)
        : group_borders_(group_borders), num_winners_(num_winning_groups), random_engine_(seed)
    {
        std::sort(group_borders_.begin(), group_borders_.end());
    }

    /**
     * @brief Call operator to divide spike messages into groups, sort them by number of spikes 
     * and return indexes of spiked neurons from the top N groups.
     * @param messages vector of spike messages.
     * @return set of indexes of spikes neurons from the top N groups.
     */
    knp::core::messaging::SpikeData operator()(const std::vector<knp::core::messaging::SpikeMessage> &messages);

private:
    std::vector<size_t> group_borders_;
    size_t num_winners_;
    std::mt19937 random_engine_;
    std::uniform_int_distribution<size_t> distribution_;
};


/**
 * @brief The KWtaPerGroup class is a definition of a message handler functor that passes 
 * further N spikes from each group.
 * @details Input messages are divided into groups.
 */
class KNP_DECLSPEC KWtaPerGroup
{
public:
    /**
     * @brief Functor constructor.
     * @details For example, we have a set of spike messages 0, 1, 2, 3, 4, 5. If `group_borders` 
     * is {2, 4}, the set of spike messages will be divided into the following groups: 
     * [0, 1], [2, 3], and [4, 5]. 
     * @param group_borders vector of spike message indexes that define right borders for 
     * each group.
     * @param winners_per_group number of spikes to pass further from each group.
     * @param seed random generator seed.
     */
    explicit KWtaPerGroup(const std::vector<size_t> &group_borders, size_t winners_per_group = 1, int seed = 0)
        : group_borders_(group_borders), winners_per_group_(winners_per_group), random_engine_(seed)
    {
        std::sort(group_borders_.begin(), group_borders_.end());
    }

    /**
     * @brief Call operator to divide spike messages into groups and return random N indexes of 
     * spiked messages from each group.
     * @param messages vector of spike messages.
     * @return set of random N indexes of spiked messages.
     */
    knp::core::messaging::SpikeData operator()(const std::vector<knp::core::messaging::SpikeMessage> &messages);

private:
    std::vector<size_t> group_borders_;
    size_t winners_per_group_;
    std::mt19937 random_engine_;
};

/**
 * @brief The SpikeUnionHandler class is a definition of a message handler functor that 
 * passes further indexes of neurons that spiked in at least one message.
 */
class KNP_DECLSPEC SpikeUnionHandler
{
public:
    /**
     * @brief Call operator to receive a vector of messages and return a set of indexes of 
     * all spiked neurons.
     * @param messages vector of spike messages.
     * @return vector of neuron indexes that spiked in at least one message.
     */
    knp::core::messaging::SpikeData operator()(const std::vector<knp::core::messaging::SpikeMessage> &messages);
};


}  // namespace knp::framework::modifier
