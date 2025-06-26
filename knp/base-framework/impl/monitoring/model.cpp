/**
 * @file model.cpp
 * @brief Functions for network construction.
 * @kaspersky_support D. Postnikov
 * @date 24.03.2025
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

#include <knp/core/messaging/messaging.h>
#include <knp/core/projection.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/model.h>
#include <knp/synapse-traits/stdp_synaptic_resource_rule.h>

#include <algorithm>
#include <map>
#include <string>
#include <tuple>

namespace knp::framework::monitoring::model
{


using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;


struct WeightByReceiverSender
{
    size_t receiver_, sender_;
    //cppcheck-suppress unusedStructMember
    float weight_;
    //cppcheck-suppress unusedStructMember
    knp::core::Step update_step_;
};


auto process_projection_weights(const knp::core::AllProjectionsVariant &proj_variant)
{
    const auto &proj = std::get<ResourceDeltaProjection>(proj_variant);
    std::vector<WeightByReceiverSender> weights_by_receiver_sender;
    for (const auto &synapse_data : proj)
    {
        float weight = std::get<knp::core::SynapseElementAccess::synapse_data>(synapse_data).rule_.synaptic_resource_;
        knp::core::Step update_step =
            std::get<knp::core::SynapseElementAccess::synapse_data>(synapse_data).rule_.last_spike_step_;
        size_t sender = std::get<knp::core::SynapseElementAccess::source_neuron_id>(synapse_data);
        size_t receiver = std::get<knp::core::SynapseElementAccess::target_neuron_id>(synapse_data);
        weights_by_receiver_sender.push_back({receiver, sender, weight, update_step});
    }
    std::sort(
        weights_by_receiver_sender.begin(), weights_by_receiver_sender.end(),
        [](const WeightByReceiverSender &v1, const WeightByReceiverSender &v2)
        {
            if (v1.receiver_ != v2.receiver_) return v1.receiver_ < v2.receiver_;
            return v1.sender_ < v2.sender_;
        });
    return weights_by_receiver_sender;
}

SpikeProcessor make_projection_weights_observer_function(
    std::ostream &weights_log, size_t period, knp::framework::ModelExecutor &model_executor, const knp::core::UID &uid)
{
    auto observer_func =
        [&weights_log, period, &model_executor, uid](const std::vector<knp::core::messaging::SpikeMessage> &)
    {
        size_t step = model_executor.get_backend()->get_step();
        std::cout << "CHECKKK observer fired on " << step << std::endl;
        if (!weights_log.good() || step % period != 0) return;
        std::cout << "CHECKKK passed check " << step << std::endl;
        // Output weights for every step that is a full square
        weights_log << "Step: " << step << std::endl;
        const auto ranges = model_executor.get_backend()->get_network_data();
        std::cout << "CHECKKK got ranges " << step << std::endl;
        for (auto &iter = *(ranges.projection_range.first); iter != *(ranges.projection_range.second); ++iter)
        {
            std::cout << "CHECKKK ranges cycle start " << step << std::endl;
            const knp::core::UID curr_proj_uid = std::visit([](const auto &proj) { return proj.get_uid(); }, *iter);
            std::cout << "CHECKKK ranges found proj uid " << step << ' ' << curr_proj_uid << std::endl;
            if (curr_proj_uid != uid) continue;
            std::cout << "CHECKKK ranges equal " << step << std::endl;
            auto weights_by_receiver_sender = process_projection_weights(*iter);
            std::cout << "CHECKKK ranges processed " << step << std::endl;
            size_t neuron = std::numeric_limits<size_t>::max();
            for (const auto &syn_data : weights_by_receiver_sender)
            {
                std::cout << "CHECKKK ranges weights start " << step << std::endl;
                size_t new_neuron = syn_data.receiver_;
                if (neuron != new_neuron)
                {
                    std::cout << "CHECKKK ranges weights output " << step << std::endl;
                    neuron = new_neuron;
                    weights_log << std::endl << "Neuron " << neuron << std::endl;
                }
                weights_log << syn_data.weight_ << "|" << syn_data.update_step_ << " ";
            }
            std::cout << "CHECKKK ranges weights end " << step << std::endl;
            weights_log << std::endl;
            return;
        }
        std::cout << "CHECKKK ranges cycle ended " << step << std::endl;
    };
    return observer_func;
}


void write_aggregated_spikes_logger_header(
    std::ostream &log_stream, const std::map<knp::core::UID, std::string> &senders_names)
{
    std::vector<std::string> vec(senders_names.size());
    std::transform(senders_names.begin(), senders_names.end(), vec.begin(), [](const auto &val) { return val.second; });
    std::sort(vec.begin(), vec.end());
    log_stream << "Index";
    for (const auto &name : vec) log_stream << ", " << name;
    log_stream << std::endl;
}


void save_aggregated_spikes_log(std::ostream &log_stream, const std::map<std::string, size_t> &values, size_t index)
{
    // Write values in order. Map is sorted by key value, that means by population name.
    log_stream << index;
    for (const auto &name_count_pair : values)
    {
        log_stream << ", " << name_count_pair.second;
    }
    log_stream << std::endl;
}


SpikeProcessor make_aggregated_spikes_observer_function(
    knp::framework::ModelExecutor &model_executor, std::ostream &log_stream, size_t period,
    const std::map<knp::core::UID, std::string> &sender_names, std::map<std::string, size_t> &accumulator)
{
    // Initialize accumulator
    accumulator.clear();
    for (const auto &val : sender_names) accumulator.insert({val.second, 0});
    auto observer_func = [&log_stream, &accumulator, &model_executor, sender_names,
                          period](const std::vector<knp::core::messaging::SpikeMessage> &messages)
    {
        size_t step = model_executor.get_backend()->get_step();
        if (step % period == 0)
        {
            // Write container to log
            save_aggregated_spikes_log(log_stream, accumulator, step);
            // Reset container
            accumulator.clear();
            for (const auto &val : sender_names) accumulator.insert({val.second, 0});
        }

        // Add spike numbers to accumulator
        for (const auto &msg : messages)
        {
            auto name_iter = sender_names.find(msg.header_.sender_uid_);
            if (name_iter == sender_names.end()) continue;
            std::string const &population_name = name_iter->second;
            accumulator[population_name] += msg.neuron_indexes_.size();
        }
    };
    return observer_func;
}


SpikeProcessor make_spikes_observer_function(
    std::ostream &log_stream, const std::map<knp::core::UID, std::string> &senders_names)
{
    auto observer_func = [&log_stream, senders_names](const std::vector<knp::core::messaging::SpikeMessage> &messages)
    {
        for (const auto &msg : messages)
        {
            const std::string name = senders_names.find(msg.header_.sender_uid_)->second;
            log_stream << "Step: " << msg.header_.send_time_ << "\nSender: " << name << std::endl;
            for (auto spike : msg.neuron_indexes_)
            {
                log_stream << spike << " ";
            }
            log_stream << std::endl;
        }
    };
    return observer_func;
}


void add_aggregated_spikes_logger(
    const knp::framework::Model &model, const std::map<knp::core::UID, std::string> &senders_names,
    knp::framework::ModelExecutor &model_executor, std::map<std::string, size_t> &spike_accumulator,
    std::ostream &log_stream, size_t logging_period)
{
    std::vector<knp::core::UID> all_senders_uids(senders_names.size());
    std::transform(
        senders_names.begin(), senders_names.end(), all_senders_uids.begin(),
        [](std::pair<knp::core::UID, std::string> const &sender) -> knp::core::UID { return sender.first; });
    write_aggregated_spikes_logger_header(log_stream, senders_names);
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_aggregated_spikes_observer_function(
            model_executor, log_stream, logging_period, senders_names, spike_accumulator),
        all_senders_uids);
}


void add_projection_weights_logger(
    std::ostream &weights_log, knp::framework::ModelExecutor &model_executor, knp::core::UID const &uid,
    size_t logging_period)
{
    std::cout << "CHECKKK added observer" << std::endl;
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_projection_weights_observer_function(weights_log, logging_period, model_executor, uid), {});
    std::cout << "CHECKKK finally added observer" << std::endl;
}


void add_spikes_logger(
    knp::framework::ModelExecutor &model_executor, const std::map<knp::core::UID, std::string> &senders_names,
    std::ostream &log_stream)
{
    std::vector<knp::core::UID> all_senders_uids(senders_names.size());
    std::transform(
        senders_names.begin(), senders_names.end(), all_senders_uids.begin(),
        [](std::pair<knp::core::UID, std::string> const &sender) -> knp::core::UID { return sender.first; });
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_spikes_observer_function(log_stream, senders_names), all_senders_uids);
}


void add_status_logger(
    knp::framework::ModelExecutor &model_executor, const knp::framework::Model &model, std::ostream &log_stream,
    size_t logging_period)
{
    std::vector<knp::core::UID> populations_uids;
    for (auto const &neuron : model.get_network().get_populations())
    {
        std::visit([&populations_uids](auto const &neuron) { populations_uids.push_back(neuron.get_uid()); }, neuron);
    }
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        [&log_stream](const std::vector<knp::core::messaging::SpikeMessage> &messages)
        {
            if (messages.empty()) return;
            log_stream << messages.size() << std::endl;
        },
        populations_uids);
}


}  //namespace knp::framework::monitoring::model
