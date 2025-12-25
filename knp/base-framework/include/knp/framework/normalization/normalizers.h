/**
 * @file normalizers.h
 * @brief Normalization functions.
 * @kaspersky_support Artiom N.
 * @date 23.07.2025
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

#pragma once

#include <knp/core/core.h>


/**
 * @brief Framework normalization namespace.
 */
namespace knp::framework::normalization
{
/**
 * @brief Type of the function to correct synapse values.
 * @tparam ValueType type of the value to correct.
 */
template <typename ValueType>
using ValueCorrector = std::function<ValueType(const ValueType &)>;


/**
 * @brief The Rescaler class is a definition of an interface used to scale values from one interval to another.
 * @tparam ValueType type of values to rescale.
 */
template <typename ValueType>
class Rescaler
{
public:
    /**
     * @brief Constructor.
     * @param in_interval_start start of the input interval.
     * @param in_interval_end end of the input interval.
     * @param out_interval_start start of the output interval.
     * @param out_interval_end end of the output interval.
     */
    Rescaler(
        ValueType in_interval_start, ValueType in_interval_end, ValueType out_interval_start,
        ValueType out_interval_end)
        : in_interval_start_(in_interval_start),
          in_interval_end_(in_interval_end),
          out_interval_start_(out_interval_start),
          out_interval_end_(out_interval_end),
          intervals_rescaler_((out_interval_end - out_interval_start) / (in_interval_end - in_interval_start))
    {
    }

    /**
     * @brief Rescale a value from the input interval to the output interval.
     * @param value value to rescale.
     * @return rescaled value.
     */
    ValueType operator()(const ValueType &value)
    {
        return out_interval_start_ + (value - in_interval_start_) * intervals_rescaler_;
    }

private:
    ValueType in_interval_start_;
    ValueType in_interval_end_;
    ValueType out_interval_start_;
    ValueType out_interval_end_;
    ValueType intervals_rescaler_;
};

}  // namespace knp::framework::normalization
