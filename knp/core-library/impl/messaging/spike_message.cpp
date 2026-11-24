/**
 * @file spike_message.cpp
 * @brief Implementation of spike message I/O operators.
 * @kaspersky_support Vartenkov A.
 * @date 23.03.2023
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

#include <spdlog/spdlog.h>

#include "spike_message_impl.h"
#include "uid_marshal.h"


namespace knp::core::messaging
{

bool operator==(const SpikeMessage &sm1, const SpikeMessage &sm2)
{
    return sm1.header_.send_time_ == sm2.header_.send_time_ && sm1.header_.sender_uid_ == sm2.header_.sender_uid_ &&
           sm1.neuron_indexes_ == sm2.neuron_indexes_;
}


std::ostream &operator<<(std::ostream &stream, const core::messaging::SpikeData &spikes)
{
    for (const auto &spike : spikes)
    {
        stream << spike << " ";
    }
    return stream;
}


std::ostream &operator<<(std::ostream &stream, const SpikeMessage &msg)
{
    stream << " " << msg.header_.sender_uid_ << " " << msg.header_.send_time_ << " " << msg.neuron_indexes_.size();
    for (auto n : msg.neuron_indexes_) stream << " " << n;
    return stream;
}


std::istream &operator>>(std::istream &stream, SpikeMessage &msg)
{
    size_t neurons_count = 0;
    stream >> msg.header_.sender_uid_ >> msg.header_.send_time_ >> neurons_count;

    if (0 == neurons_count) return stream;

    msg.neuron_indexes_.resize(neurons_count);
    for (size_t i = 0; i < neurons_count; ++i)
    {
        stream >> msg.neuron_indexes_[i];
    }
    return stream;
}


::flatbuffers::uoffset_t pack_internal(::flatbuffers::FlatBufferBuilder &builder, const SpikeMessage &msg)
{
    SPDLOG_TRACE("Packing spike message...");

    marshal::MessageHeader header(get_marshaled_uid(msg.header_.sender_uid_), msg.header_.send_time_);

    return marshal::CreateSpikeMessageDirect(builder, &header, &msg.neuron_indexes_).o;
}


std::vector<uint8_t> pack(const SpikeMessage &msg)
{
    // TODO: Don't create an instance every time.
    ::flatbuffers::FlatBufferBuilder builder;
    auto s_msg = pack_internal(builder, msg);
    marshal::FinishSpikeMessageBuffer(builder, ::flatbuffers::Offset<marshal::SpikeMessage>(s_msg));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}


SpikeMessage unpack(const marshal::SpikeMessage *s_msg)
{
    SPDLOG_TRACE("Unpacking spike message FlatBuffers class...");

    assert(s_msg);

    const marshal::MessageHeader *const s_msg_header{s_msg->header()};

    UID uid1{false};
    std::copy(
        s_msg_header->sender_uid().data()->begin(),  // clang_sa_ignore [core.CallAndMessage]
        s_msg_header->sender_uid().data()->end(),    // clang_sa_ignore [core.CallAndMessage]
        uid1.tag_.begin());

    return SpikeMessage{
        {uid1, s_msg_header->send_time()}, {s_msg->neuron_indexes()->begin(), s_msg->neuron_indexes()->end()}};
}


/*
SpikeMessage unpack(const void *buffer)
{
    SPDLOG_TRACE("Unpacking spike message buffer...");
    const marshal::SpikeMessage *const s_msg{marshal::GetSpikeMessage(buffer)};
    return unpack(s_msg);
}


SpikeMessage unpack(std::vector<uint8_t> &buffer)
{
    SPDLOG_TRACE("Unpacking spike message vector buffer...");
    return unpack(buffer.data());
}
*/

}  // namespace knp::core::messaging
