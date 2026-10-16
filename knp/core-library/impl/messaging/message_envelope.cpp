/**
 * @file message_envelope.cpp
 * @brief Message envelope implementation.
 * @kaspersky_support Artiom N.
 * @date 13.04.2023
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

#include <knp/core/messaging/message_envelope.h>
#include <knp/core/types.h>

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif
#include <knp_gen_headers/message_envelope_generated.h>
#ifdef __clang__
#    pragma clang diagnostic pop
#endif
#include <spdlog/spdlog.h>

#include "spike_message_impl.h"
#include "synaptic_impact_message_impl.h"


namespace knp::core::messaging
{

std::vector<uint8_t> pack_to_envelope(const MessageVariant &message)
{
    ::flatbuffers::FlatBufferBuilder builder;
    ::flatbuffers::Offset<marshal::MessageEnvelope> s_msg;

    SPDLOG_TRACE("Message index = {}.", message.index());

    std::visit(
        [&builder, &s_msg, &message](const auto &msg)
        {
            // Zero index is NONE.
            const auto message_type_index = message.index() + 1;
            SPDLOG_TRACE("Creating envelope for the message type {}...", message_type_index);
            s_msg = marshal::CreateMessageEnvelope(
                builder, static_cast<marshal::Message>(message_type_index), pack_internal(builder, msg));
            marshal::FinishMessageEnvelopeBuffer(builder, s_msg);
        },
        message);

    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}


MessageVariant extract_from_envelope(const void *buffer)
{
    auto msg_ev = marshal::GetMessageEnvelope(buffer);

    switch (msg_ev->message_type())
    {
        case marshal::Message_SpikeMessage:
            SPDLOG_TRACE("Unpacking spike message from the envelope...");
            return unpack(msg_ev->message_as_SpikeMessage());
        case marshal::Message_SynapticImpactMessage:
            SPDLOG_TRACE("Unpacking synaptic impact message from the envelope...");
            return unpack(msg_ev->message_as_SynapticImpactMessage());
        default:
            SPDLOG_ERROR("Unknown message type {}.", static_cast<int>(msg_ev->message_type()));
            throw std::logic_error("Unknown message type.");
    }
}

boost::mp11::mp_rename<AllMessages, std::variant> extract_from_envelope(const std::vector<uint8_t> &buffer)
{
    return extract_from_envelope(buffer.data());
}

}  // namespace knp::core::messaging
