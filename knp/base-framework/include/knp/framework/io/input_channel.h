/**
 * @file input_channel.h
 * @brief Header for input channel.
 * @kaspersky_support Vartenkov Andrey
 * @date 04.05.2023
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

#include <knp/core/core.h>

#include <utility>
#include <vector>

#include "input_converter.h"


/**
 * @brief Input channel namespace.
 */
namespace knp::framework::io::input
{

/**
 * @brief The InputChannel class is a definition of an input channel.
 */
class KNP_DECLSPEC InputChannel
{
public:
    /**
     * @brief Input channel constructor.
     * 
     * @param channel_uid sender UID to put into the message header.
     * @param endpoint endpoint used to send messages.
     * @param generator functor that generates spike messages.
     */
    InputChannel(const core::UID &channel_uid, core::MessageEndpoint &&endpoint, const DataGenerator &generator)
        : base_{channel_uid}, endpoint_(std::move(endpoint)), generator_(generator)
    {
    }

    /**
     * @copydoc InputChannel::InputChannel
     */
    InputChannel(const core::UID &channel_uid, core::MessageEndpoint &&endpoint, DataGenerator &&generator)
        : base_{channel_uid}, endpoint_(std::move(endpoint)), generator_(std::move(generator))
    {
    }

    /**
     * @brief Move constructor.
     */
    InputChannel(InputChannel &&) = default;

    /**
     * @brief Virtual default destructor of input channel.
     */
    virtual ~InputChannel() = default;

public:
    /**
     * @brief Get backend UID.
     * 
     * @return backend UID.
     */
    [[nodiscard]] const auto &get_uid() const { return base_.uid_; }
    /**
     * @brief Get tags used by the backend.
     * 
     * @return backend tag map.
     * 
     * @see TagMap.
     */
    [[nodiscard]] auto &get_tags() { return base_.tags_; }

public:
    /**
     * @brief Read data from input stream, form a spike message and send it to an endpoint.
     * 
     * @param step current step.
     * 
     * @return `true` if message was sent, `false` if no message was sent.
     * 
     * @note The method throws exceptions if an input stream is set to throw exceptions.
     */
    virtual bool send(core::Step step) { return send_data(generator_(step), step); }

protected:
    /**
     * @brief Send spike messages to message endpoint.
     * 
     * @param spikes spike messages to send.
     * @param step current step.
     * 
     * @return `true` if messages were sent, `false` otherwise.
     */
    bool send_data(const core::messaging::SpikeData &spikes, core::Step step)
    {
        if (spikes.empty())
        {
            return false;
        }

        core::messaging::SpikeMessage message{{get_uid(), step}, spikes};
        endpoint_.send_message(message);
        return true;
    }

private:
    core::BaseData base_;

    /**
     * @brief A reference to the endpoint used by channel to send messages.
     */
    core::MessageEndpoint endpoint_;

    /**
     * @brief Generator functor.
     */
    DataGenerator generator_;
};


/**
 * @brief Connect input channel to a target entity.
 * 
 * @param channel input channel.
 * @param target_endpoint endpoint to use for message exchange.
 * @param receiver_uid UID of an entity that receives spike messages.
 * 
 * @note Target entity should be able to receive spikes.
 */
inline void connect_input(
    const InputChannel &channel, core::MessageEndpoint &target_endpoint, const core::UID &receiver_uid)
{
    target_endpoint.subscribe<core::messaging::SpikeMessage>(receiver_uid, {channel.get_uid()});
}

}  // namespace knp::framework::io::input
