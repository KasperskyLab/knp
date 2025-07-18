/**
 * @file backend.h
 * @brief Class definition for backend base.
 * @kaspersky_support Artiom N.
 * @date 11.01.2023
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
#include <knp/core/device.h>
#include <knp/core/message_bus.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <atomic>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <boost/config.hpp>


/**
 * @namespace knp::backends
 * @brief Namespace for implementation of concrete backends.
 */

/**
 * @brief Core library namespace.
 */
namespace knp::core
{
/**
 * @brief The Backend class is the base class for backends.
 */
class BOOST_SYMBOL_VISIBLE Backend
{
public:
    /**
     * @brief Predicate type.
     * @details If the predicate returns `true`, network execution continues. Otherwise network execution stops.\n
     * The predicate gets a step number as a parameter.
     */
    using RunPredicate = std::function<bool(knp::core::Step)>;

public:
    /**
     * @brief Backend destructor.
     */
    virtual ~Backend();

public:
    /**
     * @brief Get backend UID.
     * @return backend UID.
     */
    [[nodiscard]] const UID &get_uid() const { return base_.uid_; }
    /**
     * @brief Get tags used by the backend.
     * @return backend tag map.
     * @see TagMap.
     */
    [[nodiscard]] auto &get_tags() { return base_.tags_; }

    /**
     * @brief Get tags used by the backend.
     * @return backend tag map.
     * @see TagMap
     */
    [[nodiscard]] const auto &get_tags() const { return base_.tags_; }

public:
    /**
     * @brief Define if plasticity is supported.
     * @return `true` if plasticity is supported, `false` if plasticity is not supported.
     */
    [[nodiscard]] virtual bool plasticity_supported() const = 0;
    /**
     * @brief Get type names of supported neurons.
     * @return vector of supported neuron type names.
     */
    [[nodiscard]] virtual std::vector<std::string> get_supported_neurons() const = 0;
    /**
     * @brief Get type names of supported synapses.
     * @return vector of supported synapse type names.
     */
    [[nodiscard]] virtual std::vector<std::string> get_supported_synapses() const = 0;
    /**
     * @brief Get indexes of supported populations.
     * @return vector of indexes of supported populations.
     */
    [[nodiscard]] virtual std::vector<size_t> get_supported_population_indexes() const = 0;
    /**
     * @brief Get indexes of supported projections.
     * @return vector of indexes of supported projections.
     */
    [[nodiscard]] virtual std::vector<size_t> get_supported_projection_indexes() const = 0;

public:
    /**
     * @brief Add projections to backend.
     * @throw exception if the `projections` parameters contains unsupported projection types.
     * @param projections projections to add.
     */
    virtual void load_all_projections(const std::vector<AllProjectionsVariant> &projections) = 0;

    /**
     * @brief Add populations to backend.
     * @throw exception if the `populations` parameter contains unsupported population types.
     * @param populations populations to add.
     */
    virtual void load_all_populations(const std::vector<AllPopulationsVariant> &populations) = 0;

    /**
     * @brief Remove projections with given UIDs from the backend.
     * @param uids UIDs of projections to remove.
     */
    virtual void remove_projections(const std::vector<UID> &uids) = 0;

    /**
     * @brief Remove populations with given UIDs from the backend.
     * @param uids UIDs of populations to remove.
     */
    virtual void remove_populations(const std::vector<UID> &uids) = 0;

public:
    /**
     * @brief Get a list of devices supported by the backend.
     * @note Constant method.
     * @return list of devices.
     * @see Device.
     */
    [[nodiscard]] virtual std::vector<std::unique_ptr<Device>> get_devices() const = 0;

    /**
     * @brief Get a list of devices on which the backend runs a network.
     * @return list of devices.
     * @see Device.
     */
    [[nodiscard]] std::vector<std::unique_ptr<Device>> &get_current_devices() { return devices_; }
    /**
     * @brief Get a list of devices on which the backend runs a network.
     * @note Constant method.
     * @return list of devices.
     * @see Device.
     */
    [[nodiscard]] const std::vector<std::unique_ptr<Device>> &get_current_devices() const { return devices_; }

    /**
     * @brief Select devices on which to run the backend.
     * @param uids set of device UIDs that the backend uses.
     */
    virtual void select_devices(const std::set<UID> &uids);

    /**
     * @brief Select devices on which to run the backend.
     * @param device selected for backend device.
     */
    virtual void select_device(std::unique_ptr<Device> &&device);

public:
    /**
     * @brief Subscribe internal endpoint to messages.
     * @details The method is used to get a subscription necessary for receiving messages of the specified type.
     * @tparam MessageType message type.
     * @param receiver receiver UID.
     * @param senders list of possible sender UIDs.
     * @return subscription.
     */
    template <typename MessageType>
    Subscription<MessageType> &subscribe(const UID &receiver, const std::vector<UID> &senders)
    {
        return message_endpoint_.subscribe<MessageType>(receiver, senders);
    }

    /**
     * @brief Get message bus used by backend.
     * @return reference to message bus.
     */
    [[nodiscard]] MessageBus &get_message_bus() { return *message_bus_; }
    /**
     * @brief Get message bus used by backend.
     * @note Constant method.
     * @return reference to message bus.
     */
    [[nodiscard]] const MessageBus &get_message_bus() const { return *message_bus_; }

    /**
     * @brief Get message endpoint.
     * @note Constant method.
     * @return message endpoint.
     */
    [[nodiscard]] virtual const core::MessageEndpoint &get_message_endpoint() const { return message_endpoint_; }
    /**
     * @brief Get message endpoint.
     * @return message endpoint.
     */
    [[nodiscard]] virtual core::MessageEndpoint &get_message_endpoint() { return message_endpoint_; }

public:
    /**
     * @brief Start network execution on the backend.
     */
    void start();
    /**
     * @brief Start network execution on the backend.
     * @param pre_step function to run before the current step.
     * @param post_step function to run after the current step.
     */
    void start(const RunPredicate &pre_step, const RunPredicate &post_step);
    /**
     * @brief Start network execution on the backend.
     * @details If the predicate returns `true`, network execution continues. Otherwise network execution stops.\n
     * The predicate gets a step number as a parameter.
     * @param run_predicate predicate function.
     */
    void start(const RunPredicate &run_predicate);

    /**
     * @brief Stop network execution on the backend.
     */
    void stop();

    /**
     * @brief Get current step.
     * @return step number.
     */
    [[nodiscard]] core::Step get_step() const { return step_; }

    /**
     * @brief Stop learning.
     */
    virtual void stop_learning() = 0;

    /**
     * @brief Restart learning.
     */
    virtual void start_learning() = 0;

public:
    /**
     * @brief Get network execution status.
     * @return `true` if network is being executed, `false` if network is not being executed.
     */
    [[nodiscard]] bool running() const { return started_; }

public:
    /**
     * @brief Initialize backend before starting network execution.
     */
    virtual void _init() = 0;

    /**
     * @brief Make one network execution step.
     * @details You can use this method for debugging purposes.
     */
    virtual void _step() = 0;

    /**
     * @brief Set backend to the uninitialized state.
     */
    virtual void _uninit();

public:
    /**
     * @brief The BaseValueIterator class is a definition of an interface to the iterator used to access populations or
     * projections by value.
     * @tparam Type one of types specified for `AllProjectionsVariant` or `AllPopulationsVariant` depending on the goal.
     */
    template <class Type>
    class BaseValueIterator
    {
    public:
        /**
         * @brief Iterator tag.
         */
        using iterator_category = std::input_iterator_tag;

        /**
         * @brief Value type.
         */
        using value_type = Type;

        /**
         * @brief Dereference a value iterator.
         * @return Copy of `Type`.
         */
        virtual Type operator*() const = 0;

        /**
         * @brief Increment an iterator.
         * @return Reference to iterator.
         */
        virtual BaseValueIterator &operator++() = 0;

        /**
         * @brief Iterator equality.
         * @param rhs another iterator.
         * @return `true` if both iterators point at the same object and have the same type.
         */
        virtual bool operator==(const BaseValueIterator &rhs) const = 0;

        /**
         * @brief Iterator inequality.
         * @param rhs another iterator.
         * @return `false` if both iterators point at the same object and have the same type.
         */
        virtual bool operator!=(const BaseValueIterator &rhs) const { return !(*this == rhs); }

        /**
         * @brief Default virtual destructor.
         */
        virtual ~BaseValueIterator() = default;
    };


    /**
     * @brief Structure used to access population and projection data.
     */
    struct DataRanges
    {
        /**
         * @brief Projection iterator pair range.
         */
        std::pair<
            std::unique_ptr<BaseValueIterator<knp::core::AllProjectionsVariant>>,
            std::unique_ptr<BaseValueIterator<knp::core::AllProjectionsVariant>>>
            projection_range;

        /**
         * @brief Population iterator pair range.
         */
        std::pair<
            std::unique_ptr<BaseValueIterator<knp::core::AllPopulationsVariant>>,
            std::unique_ptr<BaseValueIterator<knp::core::AllPopulationsVariant>>>
            population_range;
    };

    /**
     * @brief Get iterator ranges for projections and populations.
     * @return Data range for projections and populations.
     */
    [[nodiscard]] virtual DataRanges get_network_data() const = 0;

protected:
    /**
     * @brief Backend default constructor.
     */
    Backend();

    /**
     * @brief Backend constructor with custom message bus implementation.
     * @param message_bus message bus shared pointer.
     */
    explicit Backend(MessageBus &&message_bus);

    /**
     * @brief Backend constructor with custom message bus implementation.
     * @param message_bus message bus.
     */

    explicit Backend(std::shared_ptr<MessageBus> message_bus);
    /**
     * @brief Get the current step and increase the step number.
     * @return step number.
     */
    core::Step gad_step() { return step_++; }

private:
    void pre_start();

private:
    BaseData base_;
    std::atomic<bool> initialized_ = false;
    volatile std::atomic<bool> started_ = false;
    std::vector<std::unique_ptr<Device>> devices_;
    std::shared_ptr<MessageBus> message_bus_;
    MessageEndpoint message_endpoint_;
    core::Step step_ = 0;
};

}  // namespace knp::core
