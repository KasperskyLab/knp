/**
 * @file synaptic_impact_message.cpp
 * @brief Synaptic impact message I/O operator implementation.
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

#include <algorithm>

#include "synaptic_impact_message_impl.h"
#include "uid_marshal.h"


namespace knp::core::messaging
{

bool SynapticImpact::operator==(const SynapticImpact &other) const
{
    return connection_index_ == other.connection_index_ &&
           presynaptic_neuron_index_ == other.presynaptic_neuron_index_ &&
           postsynaptic_neuron_index_ == other.postsynaptic_neuron_index_ && impact_value_ == other.impact_value_;
}


bool operator==(const SynapticImpactMessage &sm1, const SynapticImpactMessage &sm2)
{
    return sm1.header_.send_time_ == sm2.header_.send_time_ && sm1.header_.sender_uid_ == sm2.header_.sender_uid_ &&
           sm1.presynaptic_population_uid_ == sm2.presynaptic_population_uid_ &&
           sm1.postsynaptic_population_uid_ == sm2.postsynaptic_population_uid_ && sm1.is_forcing_ == sm2.is_forcing_ &&
           sm1.impacts_ == sm2.impacts_;
}


std::istream &operator>>(std::istream &stream, SynapticImpact &impact)
{
    int type;
    stream >> impact.connection_index_ >> impact.impact_value_ >> type >> impact.presynaptic_neuron_index_ >>
        impact.postsynaptic_neuron_index_;
    impact.synapse_type_ = static_cast<knp::synapse_traits::OutputType>(type);
    return stream;
}


std::ostream &operator<<(std::ostream &stream, const SynapticImpact &impact)
{
    stream << impact.connection_index_ << " " << impact.impact_value_ << " " << static_cast<int>(impact.synapse_type_)
           << " " << impact.presynaptic_neuron_index_ << " " << impact.postsynaptic_neuron_index_;
    return stream;
}


std::ostream &operator<<(std::ostream &stream, const SynapticImpactMessage &msg)
{
    stream << msg.header_ << " " << msg.postsynaptic_population_uid_ << " " << msg.presynaptic_population_uid_ << " "
           << static_cast<int>(msg.is_forcing_) << " " << msg.impacts_.size();
    for (auto v : msg.impacts_) stream << " " << v;
    return stream;
}


std::istream &operator>>(std::istream &stream, SynapticImpactMessage &msg)
{
    size_t impacts_count = 0;
    int forcing_buf;
    stream >> msg.header_ >> msg.postsynaptic_population_uid_ >> msg.presynaptic_population_uid_ >> forcing_buf >>
        impacts_count;

    msg.is_forcing_ = forcing_buf;
    if (0 == impacts_count) return stream;

    msg.impacts_.resize(impacts_count);
    for (size_t i = 0; i < impacts_count; ++i)
    {
        stream >> msg.impacts_[i];
    }
    return stream;
}


::flatbuffers::uoffset_t pack_internal(::flatbuffers::FlatBufferBuilder &builder, const SynapticImpactMessage &msg)
{
    SPDLOG_TRACE("Packing synaptic impact message...");

    marshal::MessageHeader header{get_marshaled_uid(msg.header_.sender_uid_), msg.header_.send_time_};

    std::vector<knp::core::messaging::marshal::SynapticImpact> impacts;
    impacts.reserve(msg.impacts_.size());

    std::transform(
        msg.impacts_.begin(), msg.impacts_.end(), std::back_inserter(impacts),
        [](const auto &msg_val)
        {
            auto type = static_cast<knp::synapse_traits::marshal::OutputType>(msg_val.synapse_type_);
            return knp::core::messaging::marshal::SynapticImpact{
                msg_val.connection_index_, msg_val.impact_value_, type, msg_val.presynaptic_neuron_index_,
                msg_val.postsynaptic_neuron_index_};
        });

    auto pre_synaptic_uid = get_marshaled_uid(msg.presynaptic_population_uid_);
    auto post_synaptic_uid = get_marshaled_uid(msg.postsynaptic_population_uid_);

    return marshal::CreateSynapticImpactMessageDirect(
               builder, &header, &pre_synaptic_uid, &post_synaptic_uid, msg.is_forcing_, &impacts)
        .o;
}


std::vector<uint8_t> pack(const SynapticImpactMessage &msg)
{
    // TODO: Don't create instance every time.
    ::flatbuffers::FlatBufferBuilder builder;
    auto s_msg = std::move(pack_internal(builder, msg));
    marshal::FinishSynapticImpactMessageBuffer(
        builder, static_cast<::flatbuffers::Offset<marshal::SynapticImpactMessage>>(s_msg));
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}


SynapticImpactMessage unpack(const marshal::SynapticImpactMessage *s_msg)
{
    SPDLOG_TRACE("Unpacking synaptic impact message FlatBuffers class...");
    assert(s_msg);

    const marshal::MessageHeader *const s_msg_header{s_msg->header()};

    const auto &presynaptic_data = s_msg->presynaptic_population_uid()->data();
    const auto &postsynaptic_data = s_msg->postsynaptic_population_uid()->data();

    std::vector<SynapticImpact> impacts;
    impacts.reserve(s_msg->impacts()->size());
    std::transform(
        s_msg->impacts()->begin(), s_msg->impacts()->end(), std::back_inserter(impacts),
        [](const auto &msg_val)
        {
            const auto type = static_cast<knp::synapse_traits::OutputType>(msg_val->output_type());
            return SynapticImpact{
                msg_val->connection_index(), msg_val->impact_value(), type, msg_val->presynaptic_neuron_index(),
                msg_val->postsynaptic_neuron_index()};
        });
    bool is_forcing = s_msg->is_forcing();
    return SynapticImpactMessage{
        {{s_msg_header->sender_uid().data()->begin(), s_msg_header->sender_uid().data()->end()},
         s_msg_header->send_time()},
        {presynaptic_data->begin(), presynaptic_data->end()},
        {postsynaptic_data->begin(), postsynaptic_data->end()},
        is_forcing,
        std::move(impacts)};
}

}  // namespace knp::core::messaging
