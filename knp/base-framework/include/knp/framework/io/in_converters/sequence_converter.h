/**
 * @file sequence_converter.h
 * @brief Header for input sequence converter.
 * @kaspersky_support Vartenkov Andrey
 * @date 25.04.2023
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

#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/spike_message.h>
#include <knp/core/uid.h>

#include <spdlog/spdlog.h>

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


/**
 * @brief Input channel namespace.
 */
namespace knp::framework::io::input
{
/**
 * @brief The SequenceConverter class is a definition of a stream-like converter that converts a list of input values
 * into spike messages.
 * 
 * @tparam ValueType type of values received from an input stream.
 * 
 * @details For example, `SequenceConverter<float> converter{interpreter_with_threshold<float>(1.0f)}`
 * constructs a converter that interprets the input data as a spike if it is equal or greater than the threshold value.
 */
template <class ValueType>
class SequenceConverter
{
public:
    /**
     * @brief Sequence converter constructor.
     * 
     * @param interpret function that determines if the unprocessed value is a spike or not.
     * @param stream stream from which to receive data.
     * @param data_size size of input projection.
     */
    SequenceConverter(
        std::unique_ptr<std::istream> &&stream, std::function<bool(ValueType)> interpret, size_t data_size)
        : stream_(std::move(stream)), interpret_(std::move(interpret)), data_size_(data_size)
    {
    }

    /**
     * @brief Call a function that converts data from the input stream into spike messages with spiked neuron indexes.
     * 
     * @param step current step (not used in the converter).
     * 
     * @return vector of spiked neuron indexes.
     */
    core::messaging::SpikeData operator()(core::Step step = 0)
    {
        SPDLOG_TRACE("Getting message from a stream using sequence converter...");

        core::messaging::SpikeData message_data;
        for (size_t i = 0; i < data_size_; ++i)
        {
            ValueType value;
            *stream_ >> value;
            if (interpret_(value))
            {
                message_data.push_back(i);
            }
        }
        SPDLOG_TRACE("Finished loading a message, it contains {} spikes.", message_data.size());
        return message_data;
    }

public:
    /**
     * @brief Get input stream.
     * 
     * @return stream.
     */
    [[nodiscard]] std::istream &get_stream() { return *stream_; }

    /**
     * @brief Set input data size.
     * 
     * @param size input data size.
     * 
     * @details The `size` value must correspond to the size of an input projection.
     */
    void set_size(size_t size) { data_size_ = size; }

private:
    /**
     * @brief A reference to the stream from which to receive data.
     */
    std::unique_ptr<std::istream> stream_;
    /**
     * @brief Interpretation function that returns `true` if an input value is a spike, otherwise `false`. Any "wrong
     * symbol" error processing logic also goes here.
     */
    std::function<bool(ValueType)> interpret_;

    /**
     * @brief Input projection size.
     */
    size_t data_size_;
};

}  // namespace knp::framework::io::input
