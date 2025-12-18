/**
 * @file model_executor.h
 * @brief Model executor interface.
 * @kaspersky_support Artiom N.
 * @date 21.04.2023
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
#include <knp/framework/backend_loader.h>
#include <knp/framework/io/input_converter.h>
#include <knp/framework/message_handlers.h>
#include <knp/framework/model.h>
#include <knp/framework/model_loader.h>
#include <knp/framework/monitoring/observer.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{
/**
 * @brief The ModelExecutor class is a definition of an executor that runs the uploaded model on the specified backend.
 */
class KNP_DECLSPEC ModelExecutor
{
public:
    /**
     * @brief ModelExecutor constructor.
     * @param model model to run.
     * @param backend pointer to backend on which you want to run the model.
     * @param i_map input channel map.
     */
    ModelExecutor(
        knp::framework::Model &model, std::shared_ptr<core::Backend> backend, ModelLoader::InputChannelMap i_map);

    /**
     * @brief ModelExecutor destructor.
     */
    ~ModelExecutor();

public:
    /**
     * @brief Start model execution.
     */
    void start();

    /**
     * @brief Start model execution.
     * @param run_predicate predicate that stops running if the `false` value is returned.
     */
    void start(core::Backend::RunPredicate run_predicate);

    /**
     * @brief Stop model execution.
     */
    void stop();

public:
    /**
     * @brief Add observer to executor.
     * @tparam Message type of messages to observe.
     * @param message_processor functor to process received messages.
     * @param senders list of observed entities.
     */
    template <class Message>
    void add_observer(monitoring::MessageProcessor<Message> &&message_processor, const std::vector<core::UID> &senders)
    {
        observers_.emplace_back(monitoring::MessageObserver<Message>(
            get_backend()->get_message_bus().create_endpoint(), std::move(message_processor), core::UID{true}));

        std::visit([&senders](auto &entity) { entity.subscribe(senders); }, observers_.back());
    }

    /**
     * @brief Message handler function type.
     */
    using SpikeHandlerFunction =
        std::function<knp::core::messaging::SpikeData(std::vector<knp::core::messaging::SpikeMessage> &)>;

    /**
     * @brief Add spike message handler to the model executor.
     * @param message_handler_function functor to process spike messages.
     * @param senders list of entities sending messages to the handler.
     * @param receivers list of entities receiving messages from the handler.
     * @param uid handler UID.
     */
    void add_spike_message_handler(
        SpikeHandlerFunction &&message_handler_function, const std::vector<core::UID> &senders,
        const std::vector<core::UID> &receivers, const knp::core::UID &uid = knp::core::UID{});

    /**
     * @brief Unlock synapse weights.
     */
    void start_learning() { get_backend()->start_learning(); }

    /**
     * @brief Lock synapse weights.
     */
    void stop_learning() { get_backend()->stop_learning(); }

    /**
     * @brief Get pointer to backend object.
     * @return shared pointer to `Backend` object.
     */
    std::shared_ptr<core::Backend> get_backend() { return loader_.get_backend(); }

    /**
     * @brief Get message handlers vector.
     * @return Message handlers vector.
     */
    [[nodiscard]] auto const &get_message_handlers() const { return message_handlers_; }

    /**
     * @brief Get model loader object.
     * @return reference to `ModelLoader` object.
     */
    auto &get_loader() { return loader_; }

private:
    class SpikeMessageHandler;

    knp::core::BaseData base_;
    ModelLoader loader_;

    std::vector<monitoring::AnyObserverVariant> observers_;
    std::vector<std::unique_ptr<SpikeMessageHandler>> message_handlers_;
};
}  // namespace knp::framework
