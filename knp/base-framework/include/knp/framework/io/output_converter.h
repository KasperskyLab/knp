/**
 * @file output_converter.h
 * @brief Header for output converter.
 * @kaspersky_support Vartenkov Andrey
 * @date 05.04.2023
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

#include <knp/core/messaging/spike_message.h>

#include <functional>
#include <vector>


/**
 * @brief Output channel namespace.
 */
namespace knp::framework::io::output
{
/**
 * @brief Function used to convert spike messages into output data.
 * 
 * @tparam ResultType output data type.
 * 
 * @param messages list of spike messages.
 */
template <class ResultType>
using OutputConverter = std::function<ResultType(const std::vector<core::messaging::SpikeMessage> &messages)>;
}  // namespace knp::framework::io::output
