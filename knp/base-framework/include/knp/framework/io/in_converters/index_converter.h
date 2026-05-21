/**
 * @file index_converter.h
 * @brief Header for input index converter.
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

#include <knp/core/impexp.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/spike_message.h>
#include <knp/core/uid.h>

#include <spdlog/spdlog.h>

#include <functional>
#include <istream>
#include <memory>
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
 * @brief The IndexConverter class is a definition of a converter that converts lines of integers into spiked neuron
 * indexes.
 */
class KNP_DECLSPEC IndexConverter
{
public:
    /**
     * @brief Create converter with a delimiter.
     * 
     * @param stream stream from which to receive data.
     * @param delim character that delimits spiked neuron indexes.
     */
    explicit IndexConverter(std::unique_ptr<std::istream> &&stream, char delim = ',')
        : stream_(std::move(stream)), delim_(delim)
    {
    }

    /**
     * @brief Call a function that converts data from the input stream into spike messages with spiked neuron indexes.
     * 
     * @param step current step (not used in the converter).
     * 
     * @return vector of spiked neuron indexes.
     */
    core::messaging::SpikeData operator()(core::Step step = 0) const;

public:
    /**
     * @brief Get input stream.
     * 
     * @return stream.
     */
    [[nodiscard]] std::istream &get_stream() { return *stream_; }

private:
    /**
     * @brief A reference to the stream from which to receive data.
     */
    std::unique_ptr<std::istream> stream_;
    /**
     * @brief Delimiter character.
     */
    const char delim_ = ',';
};

}  // namespace knp::framework::io::input
